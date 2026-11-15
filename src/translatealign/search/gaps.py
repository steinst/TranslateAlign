"""Second pass: the gaps between the anchors.

Per gap, in this order and repeated while something changes: the one-to-n heuristic, the one-to-one
heuristic; then a length-based aligner (Gale-Church) on what is left, if the size rule allows it.

The papers leave many details open. The rules below are the ones that reproduce the output of the original
program best (found from the papers, its output, and runs of it on small inputs; its code was not read).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Sequence

from ..scoring import join
from .lengthalign import LengthAligner, galechurch_aligner

ANCHOR, HEURISTIC, LENGTH = "anchor", "heuristic", "length"


class Bead(NamedTuple):
    """Sentences of the translated (source) side aligned with sentences of the target side."""
    src: tuple
    trg: tuple
    kind: str          # "anchor" (first pass, unchanged), "heuristic" (gap heuristics), "length" (Gale-Church)


@dataclass
class GapOptions:
    max_items: int = 2                    # a sequence is 1 or 2 neighbouring items of a gap list ...
    max_sentences: int = 3                # ... of at most 3 sentences (an item can be a bead side of 2)
    source_needs_more_matches: bool = True   # a longer source side must also raise the number of matches
    target_needs_more_matches: bool = False  # the original does not ask this of the target side
    one_to_one: bool = True
    length_align: bool = True


class _B:
    __slots__ = ("src", "trg", "kind")

    def __init__(self, src, trg, kind):
        self.src, self.trg, self.kind = list(src), list(trg), kind


def size_rule(n_src: int, n_trg: int) -> bool:
    """Is a gap of this size given to the length aligner? Both sides non-empty, and the larger side less than
    twice the smaller one, or at most 3 sentences."""
    lo, hi = min(n_src, n_trg), max(n_src, n_trg)
    return lo > 0 and (hi < 2 * lo or hi <= 3)


def fill_gaps(translated: Sequence[str], target: Sequence[str], anchors: Sequence[tuple], scorer,
              length_aligner: Optional[LengthAligner] = None, options: Optional[GapOptions] = None) -> list:
    """Second pass for one document. `anchors` = (translated number, target number) pairs in rising order.

    Returns the beads in text order; sentences in no bead are unaligned.
    """
    o = options or GapOptions()
    aligner = length_aligner or galechurch_aligner
    n, m = len(translated), len(target)
    beads = [_B([i], [j], ANCHOR) for i, j in anchors]
    extra: list = []                       # beads added inside gaps (one-to-one, length aligner)
    cache: dict = {}

    def score_many(pairs):
        """Score the pairs (ss, tt) not seen before in one call; the results go into the cache."""
        todo = {}
        for ss, tt in pairs:
            key = (tuple(ss), tuple(tt))
            if key not in cache:
                todo[key] = None
        if todo:
            keys = list(todo)
            s, k = scorer.score_pairs([join([translated[i] for i in ss]) for ss, _ in keys],
                                      [join([target[j] for j in tt]) for _, tt in keys])
            for key, sv, kv in zip(keys, s, k):
                cache[key] = (float(sv), float(kv))

    def score(ss, tt):
        score_many([(ss, tt)])
        return cache[(tuple(ss), tuple(tt))]

    # The gap after the last anchor starts after that anchor's own sentences, whatever happens to the bead.
    tail = (anchors[-1][0] + 1, anchors[-1][1] + 1) if anchors else (0, 0)
    # Last sentence numbers of everything before the bead A of the gap being visited (A changes no more after
    # its gap; beads added in a gap lie before the next gap's A).
    floor_s, floor_t = -1, -1

    for g in range(len(beads) + 1):
        A = beads[g - 1] if g > 0 else None
        B = beads[g] if g < len(beads) else None
        s0 = A.src[-1] + 1 if A else 0
        t0 = A.trg[-1] + 1 if A else 0
        s1 = B.src[0] if B else n
        t1 = B.trg[0] if B else m
        if B is None and A is not None:
            s0, t0 = tail
        if s1 <= s0 and t1 <= t0:
            continue
        # The lists of the gap, fixed for this visit: bead before, the unaligned sentences, bead after.
        items = []
        for lo, hi, side in ((s0, s1, "src"), (t0, t1, "trg")):
            items.append(([list(getattr(A, side))] if A else []) + [[v] for v in range(lo, hi)]
                         + ([list(getattr(B, side))] if B else []))
        S, T = (_sequences(il, o) for il in items)
        inner: list = []                   # one-to-one beads added in this gap

        def used_by_others(bead):
            us, ut = set(), set()
            for b in (A, B, *inner):
                if b is not None and b is not bead:
                    us.update(b.src)
                    ut.update(b.trg)
            return us, ut

        def one_to_n():
            for bead in (A, B):
                if bead is None:
                    continue
                base = score(bead.src, bead.trg)
                cands = []
                for ss in S:
                    has_s = bead.src[0] in ss
                    for tt in T:
                        if ss == bead.src or tt == bead.trg or (has_s and bead.trg[0] in tt):
                            cands.append((ss, tt))
                score_many(cands)          # one call for the whole gap; the loop below keeps order and ties
                best = None
                for ss, tt in cands:
                    v = score(ss, tt)
                    if best is None or v[0] > best[0][0]:
                        best = (v, ss, tt)
                if best is None:
                    continue
                v, ss, tt = best
                # Only the best candidate is looked at. It must change exactly one side, score higher, and
                # not take a sentence of another bead; otherwise this bead stays as it is.
                if (ss == bead.src) == (tt == bead.trg) or v[0] <= base[0]:
                    continue
                us, ut = used_by_others(bead)
                if us.intersection(ss) or ut.intersection(tt):
                    continue
                # The changed bead must stay strictly between its neighbours on both sides.
                # The original program does not check this and can cross.
                if not in_order(bead, ss, tt):
                    continue
                if tt == bead.trg:
                    if o.source_needs_more_matches and v[1] <= base[1]:
                        continue
                    bead.src = list(ss)
                else:
                    if o.target_needs_more_matches and v[1] <= base[1]:
                        continue
                    bead.trg = list(tt)
                bead.kind = HEURISTIC
                return True
            return False

        def in_order(bead, ss, tt):
            row = [b for b in (A, *inner, B) if b is not None]
            i = next(k for k, b in enumerate(row) if b is bead)
            prev_s, prev_t = (row[i - 1].src[-1], row[i - 1].trg[-1]) if i > 0 else (floor_s, floor_t)
            if i + 1 < len(row):
                next_s, next_t = row[i + 1].src[0], row[i + 1].trg[0]
            elif g + 1 < len(beads):
                next_s, next_t = beads[g + 1].src[0], beads[g + 1].trg[0]
            else:
                next_s, next_t = n, m
            return prev_s < ss[0] and ss[-1] < next_s and prev_t < tt[0] and tt[-1] < next_t

        def region():
            before = [b for b in (A, *inner) if b is not None]
            lo_s = max([s0 - 1] + [b.src[-1] for b in before]) + 1
            lo_t = max([t0 - 1] + [b.trg[-1] for b in before]) + 1
            return lo_s, (B.src[0] if B else n), lo_t, (B.trg[0] if B else m)

        def one_to_one():
            a, b, c, d = region()
            if b <= a or d <= c:
                return False
            score_many([([a], [c])] + [([a], tt) for tt in T])
            v = score([a], [c])[0]
            # the first unaligned pair, if no target sequence of the gap scores higher with that sentence
            if v <= 0 or any(tt != [c] and score([a], tt)[0] > v for tt in T):
                return False
            inner.append(_B([a], [c], HEURISTIC))
            return True

        while one_to_n() or (o.one_to_one and one_to_one()):
            pass
        a, b, c, d = region()
        if o.length_align and size_rule(b - a, d - c):
            for u, w in aligner(translated[a:b], target[c:d]):
                if u and w:
                    inner.append(_B([a + i for i in u], [c + j for j in w], LENGTH))
        extra.extend(inner)
        for x in (A, *inner):
            if x is not None:
                floor_s, floor_t = max(floor_s, x.src[-1]), max(floor_t, x.trg[-1])

    out = sorted(beads + extra, key=lambda b: (b.src[0], b.trg[0]))
    return [Bead(tuple(b.src), tuple(b.trg), b.kind) for b in out]


def _sequences(item_list, o: GapOptions):
    out = []
    for k in range(1, o.max_items + 1):
        for p in range(len(item_list) - k + 1):
            flat = [v for item in item_list[p:p + k] for v in item]
            if len(flat) <= o.max_sentences:
                out.append(flat)
    return out
