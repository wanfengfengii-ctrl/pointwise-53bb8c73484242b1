# ranges — immutable wavelet-matrix range index

A pure-standard-library Python module for repeated local order-statistic
queries over a frozen column of signed 64-bit integers, with no per-query
slice copy or sort.

```python
import ranges
idx = ranges.Index([4, -2, 4, 7])
idx.kth(0, 3, 1)    # 4    (0-based 2nd smallest of [4, -2, 4])
idx.rank(1, 4, 4)   # 1    (values[-2, 4, 7] strictly < 4)
```

## Files

| file | purpose |
| --- | --- |
| `ranges.py` | the index |
| `verify.py` | differential suite against a from-scratch `sorted()` oracle |
| `demo.py` | spec examples, rank mapping, scale measurement, run manifest |
| `.pointwise-runtime.json` | real timings/results produced by `demo.py` |

Commands (both work under `python3 -I`, i.e. isolated mode):

```
python3 -I verify.py          # full suite (≈7 s, ~2.7 M checks)
python3 -I verify.py --quick  # skips the exhaustive small-n sweep
python3 -I demo.py            # measurements + rewrites .pointwise-runtime.json
python3 -I demo.py --fast     # smaller scale plan
```

## API

`ranges.Index(values)`

* `values` — iterable of Python `int` in `[-2^63, 2^63-1]`; snapshotted at
  construction, so later mutation/deletion of the caller's list has no effect.
  Non-integers (including `bool`) raise `TypeError`; integers outside the
  signed 64-bit range raise `ValueError`.

`idx.kth(l, r, k)` — the `k`-th smallest (0-based) element of `values[l:r)`,
duplicates counted individually. `0 <= l <= r <= n`, `0 <= k < r-l`.
Invalid range, invalid `k`, or non-integer arguments raise `ValueError`.

`idx.rank(l, r, x)` — count of elements in `values[l:r)` strictly `< x`.
`x` may lie outside the value domain (result is then `0` or `r-l`). Empty
interval is `0`. Invalid range or non-integer `x` raise `ValueError`.

Introspection: `len(idx)`, `idx.distinct_count`, `idx.levels`,
`idx.distinct_values()`, `idx.storage_bits()`, `idx.storage_words()`.

## Rank mapping

Construction sorts the `σ` distinct values once and assigns each a dense
rank `0 .. σ-1`:

```
distinct sorted values : v0 < v1 < ... < v(σ-1)
value -> rank         : dict lookup, used only while building
rank  -> value        : array('q') of the sorted values, O(σ) words
```

`rank(l, r, x)` maps `x` to `t = bisect_left(distinct, x)`
(= number of distinct values `< x`) and descends the matrix comparing bits
of `t`; `t == 0` returns 0 immediately and `t == σ` returns `r-l`,
so out-of-domain probes cost O(log σ) or O(1). `kth` descends by zero-counts
and converts the final rank back through the same array.

## Layout and space

A wavelet matrix over the ranks: `L = ceil(log2(σ+1))` levels, each level

* a packed bitvector of `n` bits (`array('Q')`, `ceil(n/64)` words),
* a prefix-popcount directory with `ceil(n/64)+1` entries using 32-bit
  words when `n <= 2^32-1`, 64-bit otherwise.

During the build, levels are produced one at a time; the zero/one partition
streams into the next level, so only one transient rank sequence exists and
no per-level, per-element Python object matrix is retained. The finished
index holds `L` packed buffers plus the `O(σ)` value table.

Measured retained space (see `.pointwise-runtime.json`), bitvectors + rank
directory only:

| n | σ | levels | bits | bits/element |
| --- | --- | --- | --- | --- |
| 10 000 | 64 | 6 | 90 656 | 9.06 |
| 100 000 | 1 000 | 10 | 1 500 800 | 15.01 |
| 500 000 | 50 000 | 16 | 12 001 280 | 24.00 |
| 2 000 000 | 2 000 000 | 21 | 63 006 720 | 31.50 |

That is `L * (n + 0.5n)` bit-ish ≈ 1.5 bits per element per level, i.e.
O(n log(σ+1)) bits; the value table adds O(σ) machine words (8 σ bytes
plus array headers). Note that this trades a constant factor (the 0.5×
prefix directory) for O(1) rank on packed words.

## Complexity

* Build: O(n log(σ+1) + σ log(σ+1)) expected — one hash pass for de-duplication
  and rank mapping, a Python sort of the distinct values, then `L` linear
  passes, each emitting bits and a stable partition.
* Query: O(log(σ+1)) = O(L). No interval scan or copy. This is asserted
  structurally in `verify.py::test_query_cost_is_logarithmic`: it counts
  `rank1` calls and requires exactly `2L` regardless of interval width.
* Retained: O(n log(σ+1)) bits + O(σ) machine words.

## Verification

`verify.py` checks results against an oracle that recomputes
`sorted(values[l:r])` for each query, so the index is compared to the
naive method it replaces:

* spec examples and immutability after mutating the source list;
* exhaustive sweep over all sequences of length ≤ 5 from `{-2, 0, 3, 4}`;
* randomized tuples: tiny/medium/full int64 alphabets, all `(l, r, k)` per
  small case and sampled large cases;
* 64-bit word boundaries for n ∈ {63, 64, 65, 127, 128, 129, 192, 193}, prefixes,
  suffixes, and 1–8 wide windows straddling each boundary;
* invalid arguments (ranges, `k`, types, out-of-int64 constructor values);
* space assertions (exact bit count vs. computed layout, no per-element object
	array) and the query-cost assertion above;
* a 20 000-element scale pass in dense / sparse / constant-σ regimes.

Latest run: exit 0, `PASS 2734239 checks`. The demo separately samples
large slices against `sorted()` and records the counts.
