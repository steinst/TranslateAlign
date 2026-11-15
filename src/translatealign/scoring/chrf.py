"""chrF++ (Popovic 2017): character n-grams up to 6 and word n-grams up to 2, beta = 2.

Same interface as BleuScorer. Definition as in sacrebleu 2.x (default settings, no epsilon smoothing):
character n-grams are taken from the text with whitespace removed; word n-grams from the text split on
whitespace with one leading/trailing punctuation mark separated; for each pair, precision and recall are
averaged over the orders for which both sides have n-grams, then F-beta. Lowercasing is an option
(sacrebleu's default is off; the default here is on, as for BLEU).

Direction: the score is chrF of the translation (hyps) against the target (refs). `both_directions=True`
returns the harmonic mean of the two directions (beta = 2 is recall-heavy, so the two differ).
`matches` = clipped n-gram matches summed over all orders (character and word).
"""
from __future__ import annotations

import string
from collections import Counter
from functools import lru_cache
from typing import Sequence

import numpy as np

from ._common import PreparedRefs, RefIndex, harmonic, sum_orders

_PUNCTS = set(string.punctuation)


def _words(text: str) -> list[str]:
    out: list[str] = []
    for w in text.split():
        if len(w) == 1:
            out.append(w)
        elif w[-1] in _PUNCTS:
            out += [w[:-1], w[-1]]
        elif w[0] in _PUNCTS:
            out += [w[0], w[1:]]
        else:
            out.append(w)
    return out


class ChrfScorer:
    name = "chrf"

    def __init__(self, char_order: int = 6, word_order: int = 2, beta: float = 2.0,
                 lowercase: bool = True, both_directions: bool = False):
        self.char_order, self.word_order, self.beta = char_order, word_order, beta
        self.lowercase, self.both_directions = lowercase, both_directions
        self.order = char_order + word_order
        self._ng = lru_cache(maxsize=None)(self._ngram_orders)

    def _ngram_orders(self, text: str):
        if self.lowercase:
            text = text.lower()
        chars = "".join(text.split())
        cs = [Counter(chars[i:i + n] for i in range(len(chars) - n + 1)) for n in range(1, self.char_order + 1)]
        w = _words(text)
        ws = [Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) for n in range(1, self.word_order + 1)]
        counters = cs + ws
        return counters, np.array([sum(c.values()) for c in counters], dtype=float)

    def _fscore(self, m, hn, rn, swap=False):
        """m, hn, rn: [order, ...]. Average P and R over orders where both sides have n-grams, then F-beta."""
        if swap:
            hn, rn = rn, hn
        valid = (hn > 0) & (rn > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(valid, m / np.where(hn > 0, hn, 1), 0.0)
            r = np.where(valid, m / np.where(rn > 0, rn, 1), 0.0)
        eff = np.maximum(valid.sum(axis=0), 1)
        P, R = sum_orders(p) / eff, sum_orders(r) / eff
        f2 = self.beta ** 2
        den = f2 * P + R
        return np.where(den > 0, (1 + f2) * P * R / np.where(den > 0, den, 1), 0.0)

    def _finish(self, m, hn, rn):
        f = self._fscore(m, hn, rn)
        if self.both_directions:
            f = harmonic(f, self._fscore(m, hn, rn, swap=True))
        return f, sum_orders(m)

    def score_pairs(self, hyps: Sequence[str], refs: Sequence[str]):
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs must have the same length")
        n = len(hyps)
        m = np.zeros((self.order, n)); hn = np.zeros((self.order, n)); rn = np.zeros((self.order, n))
        for i, (h, r) in enumerate(zip(hyps, refs)):
            (hc, hn[:, i]), (rc, rn[:, i]) = self._ng(h), self._ng(r)
            for k in range(self.order):
                a, b = hc[k], rc[k]
                if len(a) > len(b):
                    a, b = b, a
                m[k, i] = sum(min(c, b[g]) for g, c in a.items() if g in b)
        return self._finish(m, hn, rn)

    def prepare(self, refs: Sequence[str]) -> PreparedRefs:
        """The target side of score_matrix, prepared once (use it for several calls with the same refs)."""
        R = [self._ng(r) for r in refs]
        return PreparedRefs([RefIndex([r[0][k] for r in R]) for k in range(self.order)],
                            np.stack([r[1] for r in R], axis=1) if R else np.zeros((self.order, 0)))

    def clear_cache(self) -> None:
        """Forget the n-grams of all sentences seen (they are kept for speed; a long run needs not keep them)."""
        self._ng.cache_clear()

    def _hyp_side(self, hyps):
        H = [self._ng(h) for h in hyps]
        return [[h[0][k] for h in H] for k in range(self.order)], np.stack([h[1] for h in H], axis=1)

    def _matrix(self, enc, hn, refs: PreparedRefs):
        return self._finish(refs.matches(enc), hn[:, :, None], refs.lengths()[:, None, :])

    def score_matrix(self, hyps: Sequence[str], refs):
        """refs: a list of strings or the result of prepare() (also a slice of it)."""
        if not isinstance(refs, PreparedRefs):
            refs = self.prepare(refs) if len(refs) else refs
        if not len(hyps) or not len(refs):
            return np.zeros((len(hyps), len(refs))), np.zeros((len(hyps), len(refs)))
        counters, hn = self._hyp_side(hyps)
        return self._matrix(refs.encode(counters), hn, refs)

    def score_column_blocks(self, hyps: Sequence[str], refs: PreparedRefs, col_block: int):
        """Yield (first column, scores, matches) for blocks of `col_block` target sentences; the hypotheses
        are encoded once. Each block equals the same columns of score_matrix(hyps, refs)."""
        counters, hn = self._hyp_side(hyps)
        enc = refs.encode(counters)
        for c in range(0, len(refs), col_block):
            yield (c, *self._matrix(enc, hn, refs.slice(c, c + col_block)))
