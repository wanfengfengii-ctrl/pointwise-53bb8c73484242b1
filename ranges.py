"""ranges -- immutable wavelet-matrix index for range kth/rank queries.

The index is built from a snapshot of a sequence of signed 64-bit integers.
After construction the original sequence may be mutated or deleted without
affecting the index.

Layout (for sigma distinct values, levels = ceil(log2(sigma))):

  * one bitvector per level, packed into ``array('Q')`` words;
  * one prefix-popcount table per level (one entry per 64-bit word, plus
    sentinel), stored in ``array('I')`` or ``array('Q')``;
  * one sorted ``array('q')`` of the distinct values for mapping a query
    value to a rank and for mapping a rank back to a value.

No per-level, per-element Python object matrix is retained.  Retained space
is O(n * log(sigma + 1)) bits plus O(sigma) machine words.

Build time is O(n * log(sigma + 1) + sigma * log(sigma + 1)) expected (the
``sigma log sigma`` term is the sort of the distinct values; de-duplication
and value->rank mapping use hash tables).  Each query is O(log(sigma + 1)).
"""

from array import array
from bisect import bisect_left

__all__ = ["Index"]

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_BLOCK = 64
_BLOCK_MASK = _BLOCK - 1
_LOW_MASKS = tuple((1 << b) - 1 for b in range(_BLOCK + 1))


class _Level:
    """One wavelet-matrix level: packed bitvector + rank directory."""

    __slots__ = ("words", "prefix", "zeros", "shift")

    def __init__(self, words, prefix, zeros, shift):
        self.words = words      # array('Q'): packed bits
        self.prefix = prefix    # array('I'/'Q'): ones before each word
        self.zeros = zeros      # number of 0-bits at this level
        self.shift = shift      # bit position (MSB-first descent)

    def rank1(self, pos):
        """Number of 1-bits in [0, pos); 0 <= pos <= n."""
        word_index = pos >> 6
        offset = pos & _BLOCK_MASK
        result = self.prefix[word_index]
        if offset:
            result += (self.words[word_index] & _LOW_MASKS[offset]).bit_count()
        return result

    def storage_bits(self):
        return len(self.words) * 64 + len(self.prefix) * self.prefix.itemsize * 8


def _prefix_code(n):
    """Smallest unsigned array code able to hold a popcount up to n."""
    return "I" if n <= 0xFFFFFFFF else "Q"


def _build_level(data, shift, partition=True):
    """Split ``data`` by bit ``shift``; return (level, zeros, ones).

    The bitvector is built one 64-bit word at a time directly from the
    stream, and the same pass produces the stable zero/one partition.
    With ``partition=False`` (used for the last level) the split lists are
    skipped, since nothing consumes them.
    """
    words = array("Q")
    zeros = [] if partition else None
    ones = [] if partition else None
    append_zero = zeros.append if partition else None
    append_one = ones.append if partition else None

    word = 0
    bit = 0
    for value in data:
        if (value >> shift) & 1:
            word |= 1 << bit
            if partition:
                append_one(value)
        elif partition:
            append_zero(value)
        bit += 1
        if bit == _BLOCK:
            words.append(word)
            word = 0
            bit = 0
    if bit:
        words.append(word)

    prefix = array(_prefix_code(len(data)), [0])
    running = 0
    for word in words:
        running += word.bit_count()
        prefix.append(running)

    level = _Level(words, prefix, len(data) - running, shift)
    return level, zeros, ones


class Index:
    """Immutable range kth/rank index.

    Parameters
    ----------
    values : iterable of int
        Snapshot of signed 64-bit integers; copied during construction.
    """

    __slots__ = ("_n", "_values", "_levels", "_value_bits")

    def __init__(self, values):
        if isinstance(values, (str, bytes, bytearray)):
            raise TypeError("values must be an iterable of integers")
        try:
            snapshot = list(values)
        except TypeError:
            raise TypeError("values must be an iterable of integers") from None

        n = len(snapshot)
        for value in snapshot:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("values must contain integers")
            if value < _INT64_MIN or value > _INT64_MAX:
                raise ValueError("values must fit in signed 64-bit integers")

        self._n = n

        if n == 0:
            self._values = array("q")
            self._levels = []
            self._value_bits = 0
            return

        distinct = sorted(set(snapshot))  # O(n) built + O(sigma log sigma)
        rank_of = {value: rank for rank, value in enumerate(distinct)}
        data = [rank_of[value] for value in snapshot]

        self._values = array("q", distinct)
        sigma = len(distinct)
        self._value_bits = max(0, (sigma - 1).bit_length())

        levels = []
        shift = self._value_bits - 1
        while shift > 0:
            level, zeros, ones = _build_level(data, shift)
            levels.append(level)
            # Stable partition for the next (less significant) level.
            zeros.extend(ones)
            data = zeros
            shift -= 1
        if shift == 0:
            level, _, _ = _build_level(data, 0, partition=False)
            levels.append(level)
        self._levels = levels

    # -- introspection --------------------------------------------------

    def __len__(self):
        return self._n

    def __repr__(self):
        return (
            f"ranges.Index(n={self._n}, distinct={len(self._values)}, "
            f"levels={len(self._levels)}, bits={self.storage_bits()})"
        )

    @property
    def distinct_count(self):
        """sigma: number of distinct values in the snapshot."""
        return len(self._values)

    @property
    def levels(self):
        """Number of wavelet-matrix levels."""
        return len(self._levels)

    def distinct_values(self):
        """Return the sorted distinct values as a list (read-only copy)."""
        return list(self._values)

    def storage_bits(self):
        """Exact number of bits in the retained bitvectors + rank tables.

        The O(sigma) value table is machine words, not bits, and is reported
        separately by :meth:`storage_words`.
        """
        return sum(level.storage_bits() for level in self._levels)

    def storage_words(self):
        """Number of machine words in the retained value table (O(sigma))."""
        return len(self._values)

    # -- validation -----------------------------------------------------

    def _check_range(self, l, r):
        if type(l) is not int or type(r) is not int:
            raise ValueError("l and r must be integers")
        if l < 0 or r > self._n or l > r:
            raise ValueError(f"invalid half-open range [{l}, {r}) for n={self._n}")

    # -- queries --------------------------------------------------------

    def kth(self, l, r, k):
        """Return the k-th smallest (0-based) value in values[l:r].

        Duplicates are counted individually.  Raises ValueError for an
        invalid range or when k is outside [0, r - l).
        """
        self._check_range(l, r)
        if type(k) is not int:
            raise ValueError("k must be an integer")
        if k < 0 or k >= r - l:
            raise ValueError(f"k={k} outside [0, {r - l})")

        if not self._levels:  # sigma <= 1, all values equal
            return self._values[0]

        rank = 0
        for level in self._levels:
            ones_l = level.rank1(l)
            ones_r = level.rank1(r)
            zeros_l = l - ones_l
            zeros_r = r - ones_r
            zeros_in_range = zeros_r - zeros_l
            if k < zeros_in_range:
                l = zeros_l
                r = zeros_r
            else:
                k -= zeros_in_range
                l = level.zeros + ones_l
                r = level.zeros + ones_r
                rank |= 1 << level.shift
        return self._values[rank]

    def rank(self, l, r, x):
        """Return the number of values in values[l:r] strictly less than x.

        An empty interval always counts zero.  Raises ValueError for an
        invalid range.
        """
        self._check_range(l, r)
        if type(x) is not int:
            raise ValueError("x must be an integer")
        if l == r:
            return 0

        # Number of distinct values strictly below x; values with rank < t
        # are exactly the original values strictly below x.
        t = bisect_left(self._values, x)
        if t <= 0:
            return 0
        if t >= len(self._values):
            return r - l

        less = 0
        for level in self._levels:
            ones_l = level.rank1(l)
            ones_r = level.rank1(r)
            zeros_l = l - ones_l
            zeros_r = r - ones_r
            if (t >> level.shift) & 1:
                less += zeros_r - zeros_l
                l = level.zeros + ones_l
                r = level.zeros + ones_r
            else:
                l = zeros_l
                r = zeros_r
        return less
