#!/usr/bin/env python3
"""Demonstration, scale measurement, and run manifest for ranges.Index.

Run with:  python -I demo.py [--fast]

The demo:
  1. reproduces the examples from the specification;
  2. shows the value -> rank mapping and the retained-space accounting;
  3. runs an independent sorted-slice differential check at several scales;
  4. measures build time, memory layout, and per-query latency;
  5. executes verify.py and records everything in .pointwise-runtime.json.

Every timing is measured in this process; nothing is fabricated.
"""

import json
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ranges  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, ".pointwise-runtime.json")
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def show_examples():
    values = [4, -2, 4, 7]
    index = ranges.Index(values)
    print("== specification examples ==")
    print(f"values          : {values}")
    print(f"index           : {index!r}")
    print(f"kth(0, 3, 1)    : {index.kth(0, 3, 1)}  (expected 4)")
    print(f"rank(1, 4, 4)   : {index.rank(1, 4, 4)}  (expected 1)")
    print(f"rank(2, 2, 0)   : {index.rank(2, 2, 0)}  (empty interval)")
    print(f"distinct values : {index.distinct_values()}")
    print()
    return {
        "values": values,
        "kth(0,3,1)": index.kth(0, 3, 1),
        "rank(1,4,4)": index.rank(1, 4, 4),
        "rank(2,2,0)": index.rank(2, 2, 0),
    }


def show_rank_mapping():
    values = [9, -1, 9, 3, -1, 7]
    index = ranges.Index(values)
    distinct = index.distinct_values()
    print("== value -> rank mapping (sorted distinct values) ==")
    print(f"values     : {values}")
    print(f"distinct   : {distinct}")
    print("value -> rank : " + ", ".join(f"{v}->{i}" for i, v in enumerate(distinct)))
    print("rank -> value : " + ", ".join(f"{i}->{v}" for i, v in enumerate(distinct)))
    rows = []
    for x in (-2, -1, 0, 3, 7, 9, 10):
        rows.append({"x": x, "rank(0,n,x)": index.rank(0, len(values), x)})
    print("rank(0,n,x)   : " + ", ".join(f"x={r['x']}:{r['rank(0,n,x)']}" for r in rows))
    print()
    return {"values": values, "distinct": distinct, "ranks": rows}


def build_scale(n, sigma, rng):
    if sigma <= 1:
        values = [rng.randrange(-1000, 1000)] * n
    elif sigma <= 2:
        values = [rng.randrange(sigma) for _ in range(n)]
    else:
        values = [rng.randrange(-(1 << 62), 1 << 62) for _ in range(n)]
        # Re-map onto exactly sigma distinct values so sigma is controlled.
        distinct = sorted(set(values))[: sigma - 1]
        vals = distinct + [INT64_MAX - 1]
        values = [vals[rng.randrange(sigma)] for _ in range(n)]
    index, seconds = timed(lambda: ranges.Index(values))
    return values, index, seconds


def scale_run(n, sigma_target, query_count, sample_count, rng):
    values, index, build_seconds = build_scale(n, sigma_target, rng)
    sigma = index.distinct_count
    n = len(values)

    queries = []
    for _ in range(query_count):
        l = rng.randrange(n + 1)
        r = rng.randrange(l, n + 1)
        k = rng.randrange(r - l) if r > l else -1
        x = rng.choice(
            [INT64_MIN, INT64_MAX, 0, values[rng.randrange(n)], rng.randrange(-9, 10)]
        )
        queries.append((l, r, k, x))

    # Warm-up, then timed passes (no interval copying / sorting anywhere).
    for l, r, k, x in queries[:1000]:
        if k >= 0:
            index.kth(l, r, k)
        index.rank(l, r, x)

    def run_kth():
        checksum = 0
        for l, r, k, x in queries:
            if k >= 0:
                checksum += index.kth(l, r, k)
        return checksum

    def run_rank():
        checksum = 0
        for l, r, k, x in queries:
            checksum += index.rank(l, r, x)
        return checksum

    checksum_kth, kth_seconds = timed(run_kth)
    checksum_rank, rank_seconds = timed(run_rank)
    checksum = checksum_kth + checksum_rank
    kth_queries = sum(1 for q in queries if q[2] >= 0)
    rank_queries = len(queries)

    # Independent reference: sort each sampled slice from scratch and compare.
    sample_failures = 0
    samples = 0
    for i in range(sample_count):
        l, r, k, x = queries[rng.randrange(len(queries))]
        if k >= 0:
            want = sorted(values[l:r])[k]
            got = index.kth(l, r, k)
            samples += 1
            if want != got:
                sample_failures += 1
                print(f"  MISMATCH kth({l},{r},{k}) got {got} want {want}")
        want_rank = sum(1 for v in values[l:r] if v < x)
        got_rank = index.rank(l, r, x)
        samples += 1
        if want_rank != got_rank:
            sample_failures += 1
            print(f"  MISMATCH rank({l},{r},{x}) got {got_rank} want {want_rank}")

    kth_queries = sum(1 for q in queries if q[2] >= 0)
    rank_queries = len(queries)
    result = {
        "n": n,
        "sigma": sigma,
        "levels": index.levels,
        "build_seconds": round(build_seconds, 6),
        "build_us_per_element": round(build_seconds / n * 1e6, 3),
        "retained_bits": index.storage_bits(),
        "retained_bits_per_element": round(index.storage_bits() / n, 3),
        "value_table_words": index.storage_words(),
        "queries": len(queries),
        "kth_queries": kth_queries,
        "rank_queries": rank_queries,
        "kth_seconds": round(kth_seconds, 6),
        "kth_us_per_query": round(kth_seconds / kth_queries * 1e6, 3),
        "rank_seconds": round(rank_seconds, 6),
        "rank_us_per_query": round(rank_seconds / rank_queries * 1e6, 3),
        "queries_per_second": round(
            (kth_queries + rank_queries) / (kth_seconds + rank_seconds)
        ),
        "sample_checks": samples,
        "sample_failures": sample_failures,
        "checksum": checksum,
    }
    print(
        f"n={n:>8,}  sigma={sigma:>8,}  levels={index.levels:>2}  "
        f"build={build_seconds:7.3f}s  "
        f"space={index.storage_bits() / 1e6:8.2f} Mbit "
        f"({index.storage_bits() / n:5.2f} bit/elem)  "
        f"kth={kth_seconds / kth_queries * 1e6:6.2f} us  "
        f"rank={rank_seconds / rank_queries * 1e6:6.2f} us  "
        f"sample={samples - sample_failures}/{samples}"
    )
    return result


def run_verify():
    command = [sys.executable, "-I", os.path.join(HERE, "verify.py")]
    start = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    seconds = time.perf_counter() - start
    output = (proc.stdout or "").strip().splitlines()
    summary = output[-1] if output else ""
    record = {
        "command": "python3 -I verify.py",
        "exit_code": proc.returncode,
        "seconds": round(seconds, 3),
        "summary": summary,
    }
    if proc.stderr.strip():
        record["stderr_tail"] = proc.stderr.strip().splitlines()[-5:]
    print(f"verify.py -> exit {proc.returncode}: {summary} ({seconds:.2f}s)")
    return record


def main(argv):
    fast = "--fast" in argv
    rng = random.Random(20261115)
    started = datetime.now(timezone.utc)

    examples = show_examples()
    mapping = show_rank_mapping()

    print("== scale measurement (timed in-process, independent sorted-slice oracle) ==")
    if fast:
        plan = [(10_000, 64, 20_000, 500), (100_000, 1_000, 20_000, 500)]
    else:
        plan = [
            (10_000, 64, 100_000, 2_000),
            (100_000, 1_000, 100_000, 1_000),
            (500_000, 50_000, 100_000, 1_000),
        ]
    scale = []
    for n, sigma, queries, samples in plan:
        scale.append(scale_run(n, sigma, queries, samples, rng))
    print()

    print("== differential verification suite ==")
    verify_record = run_verify()

    manifest = {
        "schema": "pointwise-runtime/1",
        "generated_utc": started.isoformat(),
        "entry_points": {
            "index": "ranges.py",
            "verify": "verify.py",
            "demo": "demo.py",
        },
        "commands": {
            "verify": "python3 -I verify.py",
            "demo": "python3 -I demo.py",
            "demo_fast": "python3 -I demo.py --fast",
        },
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "spec_examples": examples,
        "rank_mapping": mapping,
        "verify": verify_record,
        "scale": scale,
        "notes": {
            "query_complexity": "O(log(sigma+1)) per kth/rank; no slice copy or sort",
            "build_complexity": "O(n log(sigma+1) + sigma log(sigma+1)) expected",
            "retained_space": "O(n log(sigma+1)) bits + O(sigma) machine words",
        },
    }
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"\nwrote {MANIFEST}")

    failures = sum(run["sample_failures"] for run in scale) + verify_record["exit_code"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
