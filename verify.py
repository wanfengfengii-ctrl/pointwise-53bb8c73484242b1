#!/usr/bin/env python3
"""Differential verification of ranges.Index against a naive sorted reference.

Run with:  python -I verify.py [--quick]

Every query result is compared against ``sorted(values[l:r])`` recomputed
from scratch (the naive reference the index must agree with).  The script
exits non-zero on the first mismatch and prints a summary otherwise.
"""

import itertools
import os
import random
import sys
import time

# Running under ``python -I`` suppresses the script directory on sys.path;
# add it explicitly so the sibling module is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ranges  # noqa: E402  (import after sys.path setup)

INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1

FAILURES = []
CHECKS = 0


def fail(message):
    FAILURES.append(message)
    print("FAIL:", message)


def expect_value_error(fn, label):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001 - diagnostic
        fail(f"{label}: expected ValueError, got {type(exc).__name__}: {exc}")
        return
    fail(f"{label}: expected ValueError, no exception raised")


def expect_type_error(fn, label):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except TypeError:
        return
    except Exception as exc:  # noqa: BLE001
        fail(f"{label}: expected TypeError, got {type(exc).__name__}: {exc}")
        return
    fail(f"{label}: expected TypeError, no exception raised")


def check_kth(index, values, l, r, k, label):
    global CHECKS
    CHECKS += 1
    want = sorted(values[l:r])[k]
    got = index.kth(l, r, k)
    if got != want:
        fail(f"{label}: kth({l},{r},{k}) = {got!r}, expected {want!r}")


def check_rank(index, values, l, r, x, label):
    global CHECKS
    CHECKS += 1
    want = sum(1 for v in values[l:r] if v < x)
    got = index.rank(l, r, x)
    if got != want:
        fail(f"{label}: rank({l},{r},{x}) = {got!r}, expected {want!r}")


def check_full(index, values, label):
    n = len(values)
    for l in range(n + 1):
        for r in range(l, n + 1):
            for k in range(r - l):
                check_kth(index, values, l, r, k, label)
            probes = {INT64_MIN, INT64_MAX, -7, -1, 0, 1, 5, 100}
            probes.update(values[l:r])
            for x in probes:
                check_rank(index, values, l, r, x, label)


def test_exhaustive_small():
    alphabet = (-2, 0, 3, 4)
    for n in range(0, 6):
        for values in itertools.product(alphabet, repeat=n):
            values = list(values)
            index = ranges.Index(values)
            check_full(index, values, f"exhaustive{values}")
            if FAILURES:
                return


def test_randomized():
    global CHECKS
    rng = random.Random(20261115)
    for trial in range(120):
        n = rng.randint(0, 60)
        sigma_choice = rng.choice(["tiny", "medium", "full", "extremes"])
        if sigma_choice == "tiny":
            values = [rng.randint(-1, 1) for _ in range(n)]
        elif sigma_choice == "medium":
            values = [rng.randint(-10, 10) for _ in range(n)]
        elif sigma_choice == "full":
            values = [rng.randint(INT64_MIN, INT64_MAX) for _ in range(n)]
        else:
            values = [rng.choice([INT64_MIN, INT64_MAX, 0, -1, 1]) for _ in range(n)]
        index = ranges.Index(values)
        check_full(index, values, f"random{trial}")
        # Sorted distinct view must match an independently sorted set.
        if index.distinct_values() != sorted(set(values)):
            fail(f"random{trial}: distinct_values mismatch")
        CHECKS += 1
        # Value -> rank mapping is embedded in rank()/kth(); cross-check.
        for _ in range(30):
            l = rng.randrange(n + 1)
            r = rng.randrange(l, n + 1)
            x = rng.choice(values + [INT64_MIN, INT64_MAX, 0])
            check_rank(index, values, l, r, x, f"random{trial}")
        if FAILURES:
            return


def test_word_boundaries():
    """Exercise every 64-bit word boundary: prefix/rank at all split points."""
    global CHECKS
    rng = random.Random(11)
    probes = (INT64_MIN, -100, -1, 0, 1, 2, 6, 9, 33, INT64_MAX)
    for n in (63, 64, 65, 127, 128, 129, 192, 193):
        for sigma_mode in ("dense", "sparse"):
            if sigma_mode == "dense":
                values = [rng.randrange(9) for _ in range(n)]
            else:
                values = [rng.randint(INT64_MIN, INT64_MAX) for _ in range(n)]
            index = ranges.Index(values)
            label = f"boundary-n{n}-{sigma_mode}"
            # Prefix queries [0, r) and suffix queries [l, n) for all l/r.
            for r in range(n + 1):
                for x in probes:
                    check_rank(index, values, 0, r, x, label)
            for l in range(n + 1):
                for x in probes:
                    check_rank(index, values, l, n, x, label)
            # kth over every prefix/suffix (valid k only) at a few positions.
            for r in range(1, n + 1):
                for k in (0, (r - 1) // 2, r - 1):
                    check_kth(index, values, 0, r, k, label)
            for l in range(n):
                size = n - l
                for k in (0, (size - 1) // 2, size - 1):
                    check_kth(index, values, l, n, k, label)
            # Short intervals straddling word boundaries.
            for boundary in range(64, n, 64):
                for width in (1, 2, 3, 5, 8):
                    l = max(0, boundary - width // 2)
                    r = min(n, l + width)
                    for k in range(r - l):
                        check_kth(index, values, l, r, k, label)
                    for x in probes:
                        check_rank(index, values, l, r, x, label)
            if FAILURES:
                return


def test_scale_differential():
    """Large n, sampled queries, compared against sorting each sampled slice."""
    rng = random.Random(99)
    n = 20000
    for sigma_mode in ("dense", "sparse", "singleton"):
        if sigma_mode == "dense":
            values = [rng.randint(0, 5000) for _ in range(n)]
        elif sigma_mode == "sparse":
            values = [rng.randint(INT64_MIN, INT64_MAX) for _ in range(n)]
        else:
            values = [7] * n
        index = ranges.Index(values)
        for _ in range(400):
            l = rng.randrange(n + 1)
            r = rng.randrange(l, n + 1)
            if l < r:
                k = rng.randrange(r - l)
                check_kth(index, values, l, r, k, f"scale-{sigma_mode}")
            x = rng.choice([INT64_MIN, 0, 7, rng.randint(-10, 10), INT64_MAX])
            check_rank(index, values, l, r, x, f"scale-{sigma_mode}")
        if FAILURES:
            return


def test_immutability():
    global CHECKS
    original = [4, -2, 4, 7]
    index = ranges.Index(original)
    baseline = [index.kth(0, 4, i) for i in range(4)]
    original[0] = 999
    original.append(-100)
    del original[1]
    CHECKS += 1
    after = [index.kth(0, 4, i) for i in range(4)]
    if after != baseline:
        fail(f"immutability: index changed after source mutation {baseline} -> {after}")
    CHECKS += 1
    if len(index) != 4:
        fail("immutability: length changed")
    CHECKS += 1
    if index.rank(0, 4, 4) != 1:
        fail("immutability: rank changed")


def test_spec_examples():
    global CHECKS
    index = ranges.Index([4, -2, 4, 7])
    CHECKS += 1
    if index.kth(0, 3, 1) != 4:
        fail(f"spec: kth(0,3,1) = {index.kth(0, 3, 1)}, expected 4")
    CHECKS += 1
    if index.rank(1, 4, 4) != 1:
        fail(f"spec: rank(1,4,4) = {index.rank(1, 4, 4)}, expected 1")
    CHECKS += 1
    if index.rank(2, 2, 0) != 0:
        fail("spec: empty interval rank must be 0")


def test_invalid_arguments():
    global CHECKS
    index = ranges.Index([3, 1, 2])
    n = 3
    for l, r in ((-1, 2), (0, 4), (2, 1), (4, 4), (-5, -1)):
        expect_value_error(lambda l=l, r=r: index.kth(l, r, 0), f"kth range [{l},{r})")
        expect_value_error(lambda l=l, r=r: index.rank(l, r, 0), f"rank range [{l},{r})")
    expect_value_error(lambda: index.kth(0, 0, 0), "kth empty interval")
    expect_value_error(lambda: index.kth(0, 1, 1), "kth k == length")
    expect_value_error(lambda: index.kth(0, 3, -1), "kth negative k")
    expect_value_error(lambda: index.kth(0, 3, 3), "kth k too large")
    expect_value_error(lambda: index.kth(1, 3, 3), "kth k relative to slice")

    empty = ranges.Index([])
    expect_value_error(lambda: empty.kth(0, 0, 0), "empty index kth")
    CHECKS += 1
    if empty.rank(0, 0, 123) != 0:
        fail("empty index rank should be 0")

    expect_value_error(lambda: index.kth(0.0, 2, 0), "kth float l")
    expect_value_error(lambda: index.kth(0, 2, 0.5), "kth float k")
    expect_value_error(lambda: index.rank(0, 2, "x"), "rank string x")
    expect_value_error(lambda: index.kth(True, 2, 0), "kth bool l")
    expect_value_error(lambda: index.kth(0, 2, True), "kth bool k")
    expect_type_error(lambda: ranges.Index([1.5, 2]), "non-integer value")
    expect_type_error(lambda: ranges.Index("abc"), "string values")
    expect_type_error(lambda: ranges.Index([True, False]), "boolean values")
    expect_value_error(lambda: ranges.Index([INT64_MAX + 1]), "value above int64")
    expect_value_error(lambda: ranges.Index([INT64_MIN - 1]), "value below int64")


def test_query_cost_is_logarithmic():
    """Query work must depend on log(sigma), never on the interval width."""
    global CHECKS
    rng = random.Random(7)
    values = [rng.randrange(-10**6, 10**6) for _ in range(5000)]
    index = ranges.Index(values)
    levels = index.levels
    distinct = index.distinct_values()
    probe = distinct[len(distinct) // 2]  # interior value: no early exit
    original = ranges._Level.rank1
    counter = {"calls": 0}

    def counting_rank1(self, pos):
        counter["calls"] += 1
        return original(self, pos)

    ranges._Level.rank1 = counting_rank1
    try:
        intervals = [(0, 5000), (0, 1), (2500, 2501), (100, 4000), (4999, 5000)]
        kth_counts = []
        rank_counts = []
        for l, r in intervals:
            counter["calls"] = 0
            index.kth(l, r, 0)
            kth_counts.append(counter["calls"])
            counter["calls"] = 0
            index.rank(l, r, probe)
            rank_counts.append(counter["calls"])
    finally:
        ranges._Level.rank1 = original

    CHECKS += 1
    if set(kth_counts) != {2 * levels}:
        fail(f"cost: kth rank1 calls {kth_counts}, expected {2 * levels} each")
    CHECKS += 1
    if set(rank_counts) != {2 * levels}:
        fail(f"cost: rank rank1 calls {rank_counts}, expected {2 * levels} each")


def test_storage_layout():
    global CHECKS
    rng = random.Random(5)
    for n, sigma in ((0, 0), (1, 1), (2, 2), (1000, 3), (4096, 4096), (5000, 2500)):
        values = [rng.randrange(sigma) if sigma else 0 for _ in range(n)]
        index = ranges.Index(values)
        levels = index.levels
        if sigma:
            expected_levels = max(0, (sigma - 1).bit_length())
            if levels != expected_levels:
                fail(f"storage: sigma={sigma} levels={levels} expected={expected_levels}")
        CHECKS += 1
        # Exact size: n bits packed into whole 64-bit words, plus a prefix
        # entry per word and a sentinel, per level.
        from array import array as _array

        expected_bits = 0
        for level in index._levels:
            CHECKS += 1
            if not isinstance(level.words, _array) or not isinstance(level.prefix, _array):
                fail("storage: level storage is not a packed array buffer")
                return
            words = (n + 63) // 64
            expected_bits += words * 64 + (words + 1) * level.prefix.itemsize * 8
        if index.storage_bits() != expected_bits:
            fail(
                f"storage: n={n} sigma={sigma} bits={index.storage_bits()} "
                f"expected exactly {expected_bits}"
            )
        CHECKS += 1
        # Asymptotic budget: <= log2(sigma+1) + 0.5 bits/element plus headers.
        bound = levels * (n + 64) + levels * ((n + 63) // 64 + 1) * 32 + 64 * levels
        if index.storage_bits() > bound:
            fail(
                f"storage: n={n} sigma={sigma} bits={index.storage_bits()} > bound={bound}"
            )
        CHECKS += 1
        # No Python object per (level, element) is retained anywhere.
        for level in index._levels:
            for array_like in (level.words, level.prefix):
                if not hasattr(array_like, "buffer_info"):
                    fail("storage: packed buffer expected")
                    return
        CHECKS += 1
        if hasattr(index, "_data") or hasattr(index, "_sorted_runs"):
            fail("storage: per-level element matrix retained")
    CHECKS += 1
    if ranges.Index([]).storage_bits() != 0:
        fail("storage: empty index should retain no bits")


def main(argv):
    quick = "--quick" in argv
    start = time.perf_counter()
    test_spec_examples()
    test_immutability()
    test_invalid_arguments()
    test_query_cost_is_logarithmic()
    test_storage_layout()
    test_randomized()
    test_word_boundaries()
    test_scale_differential()
    if not quick:
        test_exhaustive_small()
    elapsed = time.perf_counter() - start

    if FAILURES:
        print(f"\n{len(FAILURES)} failure(s) out of {CHECKS} checks")
        return 1
    print(f"PASS {CHECKS} checks in {elapsed:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
