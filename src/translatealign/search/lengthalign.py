"""Length-based alignment of what is left in a gap: Gale and Church (1993), through the
`galechurch` package.

A length aligner is a function (source sentences, target sentences) -> list of (source numbers, target
numbers), numbers counted from 0 within the lists it was given; either side of a pair may be empty.
"""
from __future__ import annotations

from typing import Callable, Sequence

import galechurch

LengthAligner = Callable[[Sequence[str], Sequence[str]], list]


def galechurch_aligner(source: Sequence[str], target: Sequence[str]) -> list:
    """Gale-Church on character lengths, with the parameters of the 1993 paper."""
    result = galechurch.align(list(source), list(target))
    return [(list(range(b.source_start, b.source_end)), list(range(b.target_start, b.target_end)))
            for b in result.beads]
