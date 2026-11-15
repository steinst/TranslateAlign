"""First pass (docs/SPEC.md section 3): candidates and the maximum-score monotone path.

For every translated sentence the k best-scoring target sentences are kept (score > 0); the kept pairs are
the vertices of a graph, and the path with the highest summed score in which both sentence numbers rise
strictly gives the anchors (1-to-1 beads).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def top_candidates(scores: np.ndarray, k: int = 3, row_offset: int = 0) -> list[tuple[int, int, float]]:
    """The k best target sentences of every row of a score matrix, as (row, column, score).

    Only scores above 0 count. Ties go to the lower column number.
    """
    out = []
    if scores.size == 0:
        return out
    k = min(k, scores.shape[1])
    # stable sort on the negated scores: highest first, ties in column order
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    for i in range(scores.shape[0]):
        for j in order[i]:
            v = float(scores[i, j])
            if v > 0:
                out.append((i + row_offset, int(j), v))
    return out


def candidates(scorer, translated: Sequence[str], target: Sequence[str], k: int = 3,
               block: int = 500, col_block: int = 2000) -> list[tuple[int, int, float]]:
    """Score all pairs and keep k per row, without ever holding the whole matrix.

    The matrix is computed in blocks of `block` rows by `col_block` columns (the target side is prepared once
    for the document); the k best of a row are merged over the column blocks, so the result is exactly that of
    one global sort: highest scores, ties to the lower column. Scores are kept only above 0.
    """
    prepare = getattr(scorer, "prepare", None) if hasattr(scorer, "score_column_blocks") else None
    refs = prepare(target) if prepare and len(target) else target
    m = len(target)
    out = []
    for a in range(0, len(translated), block):
        hyps = translated[a:a + block]
        best_c = best_v = None
        if prepare and m and hyps:
            blocks = ((c, sc) for c, sc, _ in scorer.score_column_blocks(hyps, refs, col_block))
        else:
            blocks = [(0, scorer.score_matrix(hyps, refs)[0])]
        for c, scores in blocks:
            if scores.size == 0:
                continue
            kk = min(k, scores.shape[1])
            order = np.argsort(-scores, axis=1, kind="stable")[:, :kk]
            vals = np.take_along_axis(scores, order, axis=1)
            cols = order + c
            if best_c is None:
                best_c, best_v = cols, vals
                continue
            # earlier columns first, so that a stable sort keeps the lower column on ties
            cols, vals = np.concatenate([best_c, cols], axis=1), np.concatenate([best_v, vals], axis=1)
            keep = np.argsort(-vals, axis=1, kind="stable")[:, :k]
            best_c, best_v = np.take_along_axis(cols, keep, axis=1), np.take_along_axis(vals, keep, axis=1)
        if best_c is None:
            continue
        for i in range(best_c.shape[0]):
            for j, v in zip(best_c[i], best_v[i]):
                if v > 0:
                    out.append((a + i, int(j), float(v)))
    return out


def best_path(cands: Sequence[tuple[int, int, float]]) -> list[tuple[int, int]]:
    """The monotone path with the highest summed score through the candidates.

    A path is a list of (row, column) pairs in which rows and columns both rise strictly. Among paths with
    the same sum the one found first wins: candidates are taken in (row, column) order, a candidate takes the
    earliest of its equally good predecessors, and the path ends at the earliest best candidate.

    Time O(V log V): candidates are processed row by row and the best path ending in a lower column is kept
    in a Fenwick tree over the columns.
    """
    cands = sorted(cands)
    if not cands:
        return []
    cols = sorted({j for _, j, _ in cands})
    pos = {j: p + 1 for p, j in enumerate(cols)}
    size = len(cols)
    tree: list = [None] * (size + 1)          # prefix maximum of (sum, -candidate number)

    def query(p):                              # best over column positions 1..p
        best = None
        while p > 0:
            if tree[p] is not None and (best is None or tree[p] > best):
                best = tree[p]
            p -= p & -p
        return best

    def update(p, value):
        while p <= size:
            if tree[p] is None or value > tree[p]:
                tree[p] = value
            p += p & -p

    total = [0.0] * len(cands)
    prev = [-1] * len(cands)
    a = 0
    while a < len(cands):
        b = a
        while b < len(cands) and cands[b][0] == cands[a][0]:
            b += 1
        for c in range(a, b):                  # all candidates of one row see only earlier rows
            q = query(pos[cands[c][1]] - 1)
            if q is not None and q[0] > 0:
                total[c], prev[c] = q[0] + cands[c][2], -q[1]
            else:
                total[c] = cands[c][2]
        for c in range(a, b):
            update(pos[cands[c][1]], (total[c], -c))
        a = b
    end = max(range(len(cands)), key=lambda c: (total[c], -c))
    path = []
    while end >= 0:
        path.append((cands[end][0], cands[end][1]))
        end = prev[end]
    return path[::-1]


def find_anchors(scorer, translated: Sequence[str], target: Sequence[str], k: int = 3) -> list[tuple[int, int]]:
    """First pass for one document: the anchors as (translated sentence number, target sentence number)."""
    return best_path(candidates(scorer, translated, target, k))
