"""Strict and lax precision, recall and F1 of an alignment against a gold alignment (docs/SPEC.md section 6).

The convention is that of the Bleualign papers, as the original program reports it:

* only beads with sentences on both sides count, in the gold alignment and in the tested one;
* strict: a tested bead is correct if the gold alignment has the identical bead;
* lax: a tested bead is correct if it overlaps a gold bead on the source side and on the target side
  (precision); a gold bead is found if a tested bead overlaps it on both sides (recall).

The strict numbers are the same as those of the Vecalign / SentAlign evaluation script; its lax recall is
defined differently (docs/RESULTS.md).

Alignment files: one bead per line, `[0, 1]:[2]` (source numbers, target numbers, counted from 0); further
`:`-separated fields (a score) are ignored.
"""
from __future__ import annotations

from ast import literal_eval
from typing import Iterable, Sequence

__all__ = ["read_alignment", "format_bead", "counts", "score", "format_scores"]


def read_alignment(path: str) -> list:
    """Read a file of `[0, 1]:[2]` lines. Returns a list of (source numbers, target numbers) tuples."""
    out = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.split(":")
            try:
                src, trg = literal_eval(fields[0].strip()), literal_eval(fields[1].strip())
                out.append((tuple(int(i) for i in src), tuple(int(j) for j in trg)))
            except (IndexError, ValueError, SyntaxError, TypeError):
                raise ValueError(f"{path}, line {number}: not an alignment line: {line.strip()!r}") from None
    return out


def format_bead(src: Iterable[int], trg: Iterable[int]) -> str:
    return f"{list(src)}:{list(trg)}"


def _both_sides(beads) -> set:
    return {(tuple(s), tuple(t)) for s, t in beads if len(s) and len(t)}


def counts(gold: Sequence, test: Sequence) -> dict:
    """Counts for one document. `gold`, `test`: beads as (source numbers, target numbers)."""
    g, t = _both_sides(gold), _both_sides(test)
    links_gold = {(i, j) for s, tr in g for i in s for j in tr}
    links_test = {(i, j) for s, tr in t for i in s for j in tr}

    def overlaps(bead, links):
        return any((i, j) in links for i in bead[0] for j in bead[1])

    return dict(test=len(t), gold=len(g), strict=len(t & g),
                lax_test=sum(overlaps(b, links_gold) for b in t),
                lax_gold=sum(overlaps(b, links_test) for b in g))


def score(gold_documents: Sequence[Sequence], test_documents: Sequence[Sequence]) -> dict:
    """Scores over a list of documents (counts are summed over the documents, then divided)."""
    if len(gold_documents) != len(test_documents):
        raise ValueError(f"{len(gold_documents)} gold documents, {len(test_documents)} tested documents")
    total = dict(test=0, gold=0, strict=0, lax_test=0, lax_gold=0)
    for g, t in zip(gold_documents, test_documents):
        for k, v in counts(g, t).items():
            total[k] += v

    def div(a, b):
        return a / b if b else 0.0

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    res = dict(total)
    res["precision_strict"], res["recall_strict"] = div(total["strict"], total["test"]), div(total["strict"], total["gold"])
    res["precision_lax"], res["recall_lax"] = div(total["lax_test"], total["test"]), div(total["lax_gold"], total["gold"])
    res["f1_strict"] = f1(res["precision_strict"], res["recall_strict"])
    res["f1_lax"] = f1(res["precision_lax"], res["recall_lax"])
    return res


def format_scores(res: dict) -> str:
    return ("             strict    lax\n"
            "precision    {precision_strict:.3f}   {precision_lax:.3f}\n"
            "recall       {recall_strict:.3f}   {recall_lax:.3f}\n"
            "F1           {f1_strict:.3f}   {f1_lax:.3f}\n"
            "({strict} of {test} beads identical to a gold bead; {gold} gold beads)").format(**res)
