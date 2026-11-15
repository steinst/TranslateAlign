"""Shared helpers: clipped n-gram matches for all pairs with sparse products."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp


def count_matrices(counters: Sequence[Counter], vocab: dict) -> sp.csr_matrix:
    """Rows = items, columns = n-gram ids (vocab is extended in place), values = counts."""
    indptr, indices, data = [0], [], []
    for c in counters:
        for g, k in c.items():
            indices.append(vocab.setdefault(g, len(vocab)))
            data.append(k)
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.array(data, dtype=np.int32), np.array(indices, dtype=np.int64), np.array(indptr)),
        shape=(len(counters), max(len(vocab), 1)),
    )


def clipped_matches(h_counters: Sequence[Counter], r_counters: Sequence[Counter]) -> np.ndarray:
    """M[i, j] = sum_g min(h_i[g], r_j[g]).

    min(a, b) = sum_{c>=1} [a >= c][b >= c], so the result is a sum of sparse 0/1 products,
    one per count level; levels above 1 are rare and very sparse.
    """
    vocab: dict = {}
    H = count_matrices(h_counters, vocab)
    R = count_matrices(r_counters, vocab)
    width = len(vocab) or 1
    H = sp.csr_matrix((H.data, H.indices, H.indptr), shape=(H.shape[0], width))
    R = sp.csr_matrix((R.data, R.indices, R.indptr), shape=(R.shape[0], width))
    out = np.zeros((H.shape[0], R.shape[0]), dtype=np.float64)
    level = 1
    while True:
        Hc = (H >= level).astype(np.float32) if level > 1 else (H > 0).astype(np.float32)
        Rc = (R >= level).astype(np.float32) if level > 1 else (R > 0).astype(np.float32)
        if Hc.nnz == 0 or Rc.nnz == 0:
            break
        out += (Hc @ Rc.T).toarray()
        level += 1
    return out


class RefIndex:
    """The reference side of one n-gram order, built once: vocabulary and the sparse count-level matrices.

    `matches(h_counters, c0, c1)` = clipped matches of the hypotheses against references c0 .. c1-1. N-grams of
    the hypotheses that no reference has cannot match and are left out.
    """

    def __init__(self, r_counters: Sequence[Counter]):
        self.vocab: dict = {}
        R = count_matrices(r_counters, self.vocab)
        self.width = max(len(self.vocab), 1)
        self.levels = []
        level = 1
        while True:
            Rc = (R >= level).astype(np.float32)
            if Rc.nnz == 0:
                break
            self.levels.append(Rc.tocsr())
            level += 1

    def encode(self, h_counters: Sequence[Counter]) -> list:
        """The hypotheses against this vocabulary: one 0/1 float32 matrix per count level."""
        get = self.vocab.get
        indptr, indices, data = [0], [], []
        for c in h_counters:
            for g, k in c.items():
                i = get(g)
                if i is not None:
                    indices.append(i)
                    data.append(k)
            indptr.append(len(indices))
        H = sp.csr_matrix((np.array(data, dtype=np.int32), np.array(indices, dtype=np.int64),
                           np.array(indptr)), shape=(len(h_counters), self.width))
        levels = []
        for level in range(1, len(self.levels) + 1):
            Hc = (H >= level).astype(np.float32)
            if Hc.nnz == 0:
                break
            levels.append(Hc)
        return levels

    def matches(self, h_levels: list, c0: int = 0, c1: int = None, n_hyps: int = 0) -> np.ndarray:
        """Clipped matches of encoded hypotheses against references c0 .. c1-1."""
        n_ref = self.levels[0].shape[0] if self.levels else 0
        c1 = n_ref if c1 is None else c1
        out = np.zeros((n_hyps, c1 - c0), dtype=np.float64)
        for Hc, Rc in zip(h_levels, self.levels):
            out += (Hc @ Rc[c0:c1].T).toarray()
        return out


class PreparedRefs:
    """The target side of a score matrix, prepared once for a whole document (a scorer's `prepare(refs)`).

    `slice(a, b)` is a view of references a .. b-1 that shares the matrices; `len()` is its size.
    """

    def __init__(self, indices: Sequence[RefIndex], lens: np.ndarray, c0: int = 0, c1: int = None):
        self.indices, self.lens = indices, lens
        self.c0 = c0
        self.c1 = lens.shape[-1] if c1 is None else c1

    def __len__(self) -> int:
        return self.c1 - self.c0

    def slice(self, a: int, b: int) -> "PreparedRefs":
        return PreparedRefs(self.indices, self.lens, self.c0 + a, self.c0 + min(b, len(self)))

    def encode(self, h_orders: Sequence[Sequence[Counter]]) -> tuple:
        """h_orders[k] = the order-k counters of the hypotheses. Encode once, then call matches() per slice."""
        return len(h_orders[0]), [ix.encode(h) for ix, h in zip(self.indices, h_orders)]

    def matches(self, encoded: tuple) -> np.ndarray:
        """-> [order, hyps, refs] clipped matches of encoded hypotheses against this slice."""
        n, per_order = encoded
        return np.stack([ix.matches(h, self.c0, self.c1, n) for ix, h in zip(self.indices, per_order)])

    def lengths(self) -> np.ndarray:
        return self.lens[..., self.c0:self.c1]


def clipped_matches_pairs(h_counters: Sequence[Counter], r_counters: Sequence[Counter]) -> np.ndarray:
    """Matches for aligned lists of pairs (h_i, r_i)."""
    out = np.zeros(len(h_counters))
    for i, (h, r) in enumerate(zip(h_counters, r_counters)):
        if len(h) > len(r):
            h, r = r, h
        out[i] = sum(min(k, r[g]) for g, k in h.items() if g in r)
    return out


def sum_orders(a: np.ndarray) -> np.ndarray:
    """Sum over the first axis, one term after the other.

    `a.sum(axis=0)` adds in another order (pairwise) when the array is laid out so that the first axis is the
    fastest one (e.g. one pair at a time), so the same pair could get a score that differs in the last digit
    from the one it gets inside a matrix, and a tie could break the other way. This does not depend on the shape.
    """
    out = np.array(a[0], dtype=float)
    for x in a[1:]:
        out += x
    return out


def harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    s = a + b
    return np.where(s > 0, 2 * a * b / np.where(s > 0, s, 1), 0.0)
