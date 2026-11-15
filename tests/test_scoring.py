import random

import numpy as np
import pytest

from translatealign.scoring import BleuScorer, ChrfScorer, join, tokenize

sacrebleu = pytest.importorskip("sacrebleu")

SENTS = [
    "The cat sat on the mat .", "A cat sat on a mat .", "Hello world", "Hello , world !",
    "completely different text here", "Il est venu hier soir , mais il est reparti .",
    "yes", "Yes .", "", "the the the the cat", "the cat the cat the dog", "Der Hund bellt laut .",
    "(He said) hello, and left.", "It's a well-known fact that state-of-the-art isn't free.",
]


def _random_sentences(n, seed=1):
    rnd = random.Random(seed)
    vocab = [f"w{i}" for i in range(40)] + [",", "."]
    return [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, 25))) for _ in range(n)]


def test_tokenize():
    assert tokenize("Hello, World! It's a well-known fact.") == [
        "hello", ",", "world", "!", "it's", "a", "well-known", "fact", "."]


def test_bleu_hand_computed():
    # 7 tokens each; 6 unigram matches, 4 bigram matches: sqrt(6/7 * 4/6)
    s, m = BleuScorer().score_pairs(["the cat sat on the mat ."], ["the cat sat on a mat ."])
    assert s[0] == pytest.approx((6 / 7 * 4 / 6) ** 0.5)
    assert m[0] == 10


def test_bleu_unrelated_is_zero_and_identical_is_one():
    s, _ = BleuScorer().score_pairs(["a b c d", "a b c d"], ["x y z w", "a b c d"])
    assert s[0] == 0 and s[1] == pytest.approx(1)


def test_bleu_no_common_bigram_is_zero():
    s, m = BleuScorer().score_pairs(["a b c"], ["c b a"])
    assert s[0] == 0 and m[0] == 3  # unigrams match, no smoothing


def test_bleu_brevity_penalty_symmetric_score():
    b = BleuScorer()
    s1, _ = b.score_pairs(["a b c"], ["a b c d e f"])
    s2, _ = b.score_pairs(["a b c d e f"], ["a b c"])
    assert s1[0] == pytest.approx(s2[0]) and 0 < s1[0] < 1


def test_bleu_match_count_unigram():
    _, m = BleuScorer(match_count="unigram").score_pairs(["the cat sat on the mat ."], ["the cat sat on a mat ."])
    assert m[0] == 6


def test_bleu_clipping():
    _, m = BleuScorer(n=1).score_pairs(["the the the"], ["the cat"])
    assert m[0] == 1


def test_one_word_sentence_strict_vs_effective():
    assert BleuScorer(effective_order=True).score_pairs(["hello"], ["hello"])[0][0] == pytest.approx(1)
    assert BleuScorer(effective_order=False).score_pairs(["hello"], ["hello"])[0][0] == 0


@pytest.mark.parametrize("scorer", [BleuScorer(), BleuScorer(n=3), BleuScorer(effective_order=False),
                                    BleuScorer(match_count="unigram"), ChrfScorer(),
                                    ChrfScorer(both_directions=True), ChrfScorer(char_order=3, word_order=0)])
def test_matrix_equals_pairwise(scorer):
    hyps, refs = _random_sentences(30, 1) + SENTS, _random_sentences(25, 2) + SENTS
    S, M = scorer.score_matrix(hyps, refs)
    assert S.shape == (len(hyps), len(refs))
    ii, jj = np.meshgrid(range(len(hyps)), range(len(refs)), indexing="ij")
    s, m = scorer.score_pairs([hyps[i] for i in ii.ravel()], [refs[j] for j in jj.ravel()])
    np.testing.assert_allclose(S.ravel(), s, atol=1e-12)
    np.testing.assert_array_equal(M.ravel(), m)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_bleu_equals_sacrebleu_one_direction(n):
    """Same definition as sacrebleu sentence BLEU with smooth_method='none' on pre-tokenised input
    (effective order, geometric mean, brevity penalty). Differs from it only by the harmonic mean over
    both directions, which sacrebleu does not have."""
    sb = sacrebleu.metrics.BLEU(max_ngram_order=n, smooth_method="none", tokenize="none", effective_order=True)
    ours = BleuScorer(n=n)
    for h in SENTS[:12] + _random_sentences(40):
        for r in SENTS[:12] + _random_sentences(10, 5):
            th, tr = " ".join(tokenize(h)), " ".join(tokenize(r))
            if not th or not tr:
                continue
            assert ours.directional(h, r) == pytest.approx(sb.sentence_score(th, [tr]).score / 100, abs=1e-9), (h, r)


def test_bleu_harmonic_mean_of_two_sacrebleu_scores():
    sb = sacrebleu.metrics.BLEU(max_ngram_order=2, smooth_method="none", tokenize="none", effective_order=True)
    h, r = "the cat sat on the mat today .", "a cat sat on a mat ."
    a = sb.sentence_score(" ".join(tokenize(h)), [" ".join(tokenize(r))]).score / 100
    b = sb.sentence_score(" ".join(tokenize(r)), [" ".join(tokenize(h))]).score / 100
    s, _ = BleuScorer().score_pairs([h], [r])
    assert s[0] == pytest.approx(2 * a * b / (a + b))


@pytest.mark.parametrize("word_order", [0, 2])
def test_chrf_equals_sacrebleu(word_order):
    sc = sacrebleu.metrics.CHRF(word_order=word_order, lowercase=True)
    ours = ChrfScorer(word_order=word_order)
    hyps, refs = _random_sentences(30, 3) + SENTS, _random_sentences(20, 4) + SENTS
    S, _ = ours.score_matrix(hyps, refs)
    for i, h in enumerate(hyps):
        for j, r in enumerate(refs):
            if not h.strip() or not r.strip():
                continue
            assert S[i, j] == pytest.approx(sc.sentence_score(h, [r]).score / 100, abs=1e-9), (h, r)


def test_chrf_both_directions_is_harmonic_mean():
    h, r = "the cat sat on the mat today .", "a cat sat on a mat ."
    one = ChrfScorer().score_pairs([h], [r])[0][0]
    other = ChrfScorer().score_pairs([r], [h])[0][0]
    both = ChrfScorer(both_directions=True).score_pairs([h], [r])[0][0]
    assert both == pytest.approx(2 * one * other / (one + other))


def test_empty_inputs():
    for sc in (BleuScorer(), ChrfScorer()):
        assert sc.score_matrix([], ["a"])[0].shape == (0, 1)
        assert sc.score_pairs([""], ["a b"])[0][0] == 0
        assert sc.score_pairs([], [])[0].shape == (0,)


def test_joined_sentences():
    b = BleuScorer()
    s_one, _ = b.score_pairs(["the cat sat ."], ["the cat sat ."])
    s_join, m = b.score_pairs([join(["the cat sat .", "it purred ."])], [join(["the cat sat .", "it purred ."])])
    assert s_one[0] == pytest.approx(1) and s_join[0] == pytest.approx(1) and m[0] == 13
