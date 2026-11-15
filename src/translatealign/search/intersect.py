"""Several runs: keep only the beads that are identical in every run."""
from __future__ import annotations

from typing import Sequence

from .gaps import Bead


def flip(beads: Sequence[Bead]) -> list:
    """Beads of a run in the other direction (target translated, aligned with the source), turned round so
    that `src` is the source side again."""
    return sorted((Bead(b.trg, b.src, b.kind) for b in beads), key=lambda b: (b.src[0], b.trg[0]))


def intersect(*runs: Sequence[Bead]) -> list:
    """The beads (same source sentences, same target sentences) that occur in all runs, in text order.
    The kind is taken from the first run."""
    if not runs:
        return []
    common = set((b.src, b.trg) for b in runs[0])
    for run in runs[1:]:
        common &= set((b.src, b.trg) for b in run)
    return [b for b in runs[0] if (b.src, b.trg) in common]
