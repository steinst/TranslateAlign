"""Modified BLEU of Sennrich & Volk (2010): n-grams up to n (default 2), tokenised and lowercased,
computed in both directions and combined by the harmonic mean.

Interface shared with ChrfScorer:
    score_pairs(hyps, refs)  -> (scores, matches)   1-D arrays, item i against item i
    score_matrix(hyps, refs) -> (scores, matches)   2-D arrays [len(hyps), len(refs)]
`matches` = number of n-gram matches (clipped) counted over the orders given by `match_count`.
Scores are in [0, 1]. Callers join sentences themselves (translatealign.scoring.join).
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np

from ._common import PreparedRefs, RefIndex, harmonic, sum_orders

_TOKEN = re.compile(r"\w+(?:['’\-]\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Default tokeniser: lowercase; words (with inner apostrophes/hyphens) and single punctuation marks."""
    return _TOKEN.findall(text.lower())


_13A = [(re.compile(r"([\{-\~\[-\` -\&\(-\+\:-\@\/])"), r" \1 "),   # punctuation, except . , - '
        (re.compile(r"([^0-9])([\.,])"), r"\1 \2 "),                         # . and , unless after a digit
        (re.compile(r"([\.,])([^0-9])"), r" \1 \2"),                         # . and , unless before a digit
        (re.compile(r"([0-9])(-)"), r"\1 \2 ")]                              # dash after a digit


def tokenize_13a(text: str) -> list[str]:
    """The tokenisation rules of the NIST script mteval-v13a (the usual BLEU tokenisation), then lowercase.

    This is what reproduces the scores of the original Bleualign on its evaluation files.
    Written from the published rules; tests compare it with sacrebleu's implementation.
    """
    text = text.replace("<skipped>", "").replace("-\n", "").replace("\n", " ")
    if "&" in text:
        text = text.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = " " + text + " "
    for rx, rep in _13A:
        text = rx.sub(rep, text)
    return text.lower().split()


def _ngrams(tokens: Sequence[str], k: int) -> Counter:
    return Counter(zip(*(tokens[i:] for i in range(k))))


class BleuScorer:
    """
    n:            highest n-gram order (paper: 2).
    tokenizer:    str -> list[str]; it must lowercase if lowercasing is wanted (the default does).
    match_count:  "all" = clipped matches summed over orders 1..n; "unigram" = order 1 only.
    effective_order: if True a hypothesis too short for order k (no k-grams) is scored with the lower orders
                  only (as sacrebleu's sentence BLEU); if False its score is 0. (The reference having no
                  k-grams makes the match count 0 at that order, hence the score 0, in both modes.)
    No smoothing: a pair with no match at some order scores 0.
    """

    name = "bleu"

    def __init__(self, n: int = 2, tokenizer: Callable[[str], list[str]] = tokenize,
                 match_count: str = "all", effective_order: bool = True):
        if match_count not in ("all", "unigram"):
            raise ValueError("match_count must be 'all' or 'unigram'")
        self.n, self.tokenizer = n, tokenizer
        self.match_count, self.effective_order = match_count, effective_order
        self._tok = lru_cache(maxsize=None)(lambda s: tuple(self.tokenizer(s)))
        self._ng = lru_cache(maxsize=None)(self._ngram_orders)

    @classmethod
    def reference(cls, **kwargs) -> "BleuScorer":
        """The settings that reproduce the original Bleualign: 1- and 2-grams, mteval-13a
        tokens, lowercased, a pair without a common bigram (or with a one-token side) scores 0."""
        opts = dict(n=2, tokenizer=tokenize_13a, effective_order=False)
        opts.update(kwargs)
        return cls(**opts)

    def _ngram_orders(self, text: str):
        t = self._tok(text)
        return tuple(_ngrams(t, k) for k in range(1, self.n + 1)), len(t)

    def _direction(self, m: np.ndarray, hl, rl) -> np.ndarray:
        """BLEU of one direction: m [n, ...] matches per order, hl/rl token counts of hypothesis/reference."""
        k = np.arange(1, self.n + 1).reshape((-1,) + (1,) * m[0].ndim)
        hn = np.maximum(hl - k + 1, 0)           # number of k-grams of each side
        rn = np.maximum(rl - k + 1, 0)
        has = hn > 0                              # as sacrebleu: the order counts if the hypothesis has k-grams
        valid = has if self.effective_order else np.ones_like(has)
        zero = (valid & (m == 0)).any(axis=0) | ~valid.any(axis=0)
        if not self.effective_order:
            zero |= (~has).any(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.where(valid & (m > 0), np.log(np.where(hn > 0, m, 1) / np.maximum(hn, 1)), 0.0)
            bp = np.where(hl >= rl, 1.0, np.exp(1 - rl / np.maximum(hl, 1)))
        score = bp * np.exp(sum_orders(logp) / np.maximum(valid.sum(axis=0), 1))
        return np.where(zero, 0.0, score)

    def _combine(self, m: np.ndarray, hl, rl):
        """Harmonic mean of both directions, and the match count."""
        a = self._direction(m, hl, rl)
        b = self._direction(m, rl, hl)
        matches = sum_orders(m) if self.match_count == "all" else m[0]
        return harmonic(a, b), matches

    def directional(self, hyp: str, ref: str) -> float:
        """BLEU of one direction only (hyp against ref), for comparison with sacrebleu."""
        (ms,), (hl, rl) = self._pair_stats([hyp], [ref])
        return float(self._direction(ms, hl, rl)[0])

    def _pair_stats(self, hyps, refs):
        ms = np.zeros((self.n, len(hyps)))
        hl = np.zeros(len(hyps)); rl = np.zeros(len(hyps))
        for i, (h, r) in enumerate(zip(hyps, refs)):
            (hg, hl[i]), (rg, rl[i]) = self._ng(h), self._ng(r)
            for k in range(self.n):
                a, b = hg[k], rg[k]
                if len(a) > len(b):
                    a, b = b, a
                ms[k, i] = sum(min(c, b[g]) for g, c in a.items() if g in b)
        return (ms,), (hl, rl)

    def score_pairs(self, hyps: Sequence[str], refs: Sequence[str]):
        if len(hyps) != len(refs):
            raise ValueError("hyps and refs must have the same length")
        if not len(hyps):
            return np.zeros(0), np.zeros(0)
        (ms,), (hl, rl) = self._pair_stats(hyps, refs)
        return self._combine(ms, hl, rl)

    def prepare(self, refs: Sequence[str]) -> PreparedRefs:
        """The target side of score_matrix, prepared once (use it for several calls with the same refs)."""
        R = [self._ng(r) for r in refs]
        return PreparedRefs([RefIndex([r[0][k] for r in R]) for k in range(self.n)],
                            np.array([r[1] for r in R], dtype=float))

    def clear_cache(self) -> None:
        """Forget the n-grams of all sentences seen (they are kept for speed; a long run needs not keep them)."""
        self._ng.cache_clear()
        self._tok.cache_clear()

    def _hyp_side(self, hyps):
        H = [self._ng(h) for h in hyps]
        return [[h[0][k] for h in H] for k in range(self.n)], np.array([h[1] for h in H], dtype=float)

    def _matrix(self, enc, hl, refs: PreparedRefs):
        return self._combine(refs.matches(enc), hl[:, None], refs.lengths()[None, :])

    def score_matrix(self, hyps: Sequence[str], refs):
        """refs: a list of strings or the result of prepare() (also a slice of it)."""
        if not isinstance(refs, PreparedRefs):
            refs = self.prepare(refs) if len(refs) else refs
        if not len(hyps) or not len(refs):
            return np.zeros((len(hyps), len(refs))), np.zeros((len(hyps), len(refs)))
        counters, hl = self._hyp_side(hyps)
        return self._matrix(refs.encode(counters), hl, refs)

    def score_column_blocks(self, hyps: Sequence[str], refs: PreparedRefs, col_block: int):
        """Yield (first column, scores, matches) for blocks of `col_block` target sentences; the hypotheses
        are encoded once. Each block equals the same columns of score_matrix(hyps, refs)."""
        counters, hl = self._hyp_side(hyps)
        enc = refs.encode(counters)
        for c in range(0, len(refs), col_block):
            yield (c, *self._matrix(enc, hl, refs.slice(c, c + col_block)))
