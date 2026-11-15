"""The search of Sennrich & Volk (2010), re-implemented from the papers.

    align(translated, target)              one document -> beads
    align_documents(translated, target)    files with hard delimiters -> beads with line numbers
"""
from __future__ import annotations

from typing import Optional, Sequence

from ..scoring import BleuScorer
from .anchors import best_path, candidates, find_anchors, top_candidates
from .gaps import ANCHOR, HEURISTIC, LENGTH, Bead, GapOptions, fill_gaps, size_rule
from .intersect import flip, intersect
from .lengthalign import galechurch_aligner

__all__ = ["align", "align_documents", "split_documents", "Bead", "GapOptions", "find_anchors", "fill_gaps",
           "best_path", "candidates", "top_candidates", "size_rule", "intersect", "flip",
           "galechurch_aligner", "ANCHOR", "HEURISTIC", "LENGTH"]

DELIMITER = ".EOA"


def align(translated: Sequence[str], target: Sequence[str], scorer=None, *, k: int = 3,
          length_aligner=None, options: Optional[GapOptions] = None) -> list:
    """Align one document.

    translated: the machine translation of the source text, one item per source sentence.
    target:     the sentences of the target text.
    scorer:     a scorer from translatealign.scoring (default: BLEU with the settings of the original).
    Returns beads (src numbers, trg numbers, kind) in text order; numbers count from 0.
    """
    scorer = scorer or BleuScorer.reference()
    anchors = find_anchors(scorer, translated, target, k)
    beads = fill_gaps(translated, target, anchors, scorer, length_aligner, options)
    clear = getattr(scorer, "clear_cache", None)
    if clear:
        clear()        # the n-grams of this document are not needed for the next one
    return beads


def split_documents(lines: Sequence[str], delimiter: str = DELIMITER, like: Optional[Sequence[str]] = None):
    """Split lines at hard delimiters. Returns a list of (first line number, lines) per document.

    `like`: take the delimiter positions from these lines instead (a translation is split where the text it
    translates has its delimiters, because the delimiter line itself may have been translated).
    """
    marks = like if like is not None else lines
    if len(marks) != len(lines):
        raise ValueError("the translation must have as many lines as the text it translates")
    docs, start, cur = [], 0, []
    for number, (line, mark) in enumerate(zip(lines, marks)):
        if mark.strip() == delimiter:
            docs.append((start, cur))
            start, cur = number + 1, []
        else:
            cur.append(line)
    docs.append((start, cur))
    return docs


def align_documents(translated: Sequence[str], target: Sequence[str], scorer=None, *,
                    source: Optional[Sequence[str]] = None, delimiter: str = DELIMITER, **kwargs) -> list:
    """Align texts with hard delimiter lines; the search never crosses a delimiter.

    source: the untranslated source lines, if the delimiter lines of `translated` may have been changed by
    the translation. Returns beads whose numbers are line numbers in the files given.
    """
    src_docs = split_documents(translated, delimiter, like=source)
    trg_docs = split_documents(target, delimiter)
    if len(src_docs) != len(trg_docs):
        raise ValueError(f"{len(src_docs)} documents on the source side, {len(trg_docs)} on the target side")
    scorer = scorer or BleuScorer.reference()
    out = []
    for (so, sl), (to, tl) in zip(src_docs, trg_docs):
        for b in align(sl, tl, scorer, **kwargs):
            out.append(Bead(tuple(i + so for i in b.src), tuple(j + to for j in b.trg), b.kind))
    return out
