import hashlib
import itertools
import json
import os
import random

import pytest

from reference_data import HERE, RUNS, load_run, reference
from translatealign.scoring import BleuScorer, ChrfScorer, tokenize_13a
from translatealign.search import (ANCHOR, HEURISTIC, LENGTH, Bead, align, align_documents, best_path,
                                   find_anchors, galechurch_aligner, flip, intersect, size_rule, split_documents,
                                   top_candidates)

REPO = os.environ.get("TRANSLATEALIGN_BLEUALIGN_REPO")
needs_repo = pytest.mark.skipif(not REPO, reason="set TRANSLATEALIGN_BLEUALIGN_REPO to a clone of Bleualign")


def W(a, b, p="w"):
    return " ".join(f"{p}{i}" for i in range(a, b))


PRE, POST = W(0, 10, "p"), W(0, 10, "q")


def pairs(beads):
    return [(list(b.src), list(b.trg)) for b in beads]


def no_length_aligner(src, trg):
    return []


# ---------------------------------------------------------------- tokeniser

def test_tokenize_13a_equals_sacrebleu():
    sacrebleu = pytest.importorskip("sacrebleu")
    from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
    t = Tokenizer13a()
    for s in ["a_b x-y 3-4 1,000.5 End. (x) [y] {z} it's &amp; &lt;b&gt; 5,a a,5 42.- !) «q»", "", " . ",
              "Les Alpes , 1956 , p. 83 ) .", "fr. 42.- suisse", "<seil ! )"]:
        assert tokenize_13a(s) == t(s).lower().split()


# ---------------------------------------------------------------- first pass

def _quadratic_path(cands):
    """The same search written the slow, obvious way."""
    cands = sorted(cands)
    total, prev = [], []
    for a, (i, j, v) in enumerate(cands):
        best, p = 0.0, -1
        for c in range(a):
            if cands[c][0] < i and cands[c][1] < j and total[c] > best:
                best, p = total[c], c
        total.append(best + v)
        prev.append(p)
    if not cands:
        return []
    end = max(range(len(cands)), key=lambda c: (total[c], -c))
    out = []
    while end >= 0:
        out.append(cands[end][:2])
        end = prev[end]
    return out[::-1]


def _exhaustive_best(cands):
    best = 0.0
    for r in range(1, len(cands) + 1):
        for sub in itertools.combinations(sorted(cands), r):
            if all(a[0] < b[0] and a[1] < b[1] for a, b in zip(sub, sub[1:])):
                best = max(best, sum(c[2] for c in sub))
    return best


def test_best_path_against_exhaustive_search():
    rnd = random.Random(3)
    for _ in range(200):
        n = rnd.randint(0, 11)
        cands = list({(rnd.randint(0, 5), rnd.randint(0, 5)): 0 for _ in range(n)})
        cands = [(i, j, rnd.choice([0.1, 0.2, 0.25, 0.5, 1.0])) for i, j in cands]
        path = best_path(cands)
        assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(path, path[1:]))
        score = {(i, j): v for i, j, v in cands}
        assert sum(score[p] for p in path) == pytest.approx(_exhaustive_best(cands))


def test_best_path_equals_quadratic_version_with_ties():
    rnd = random.Random(5)
    for _ in range(100):
        cands = list({(rnd.randint(0, 40), rnd.randint(0, 40)) for _ in range(rnd.randint(1, 120))})
        cands = [(i, j, rnd.choice([0.25, 0.5, 0.5, 1.0])) for i, j in cands]   # many equal sums
        assert best_path(cands) == _quadratic_path(cands)


def test_top_candidates_keeps_k_positive_per_row():
    import numpy as np
    m = np.array([[0.1, 0.5, 0.5, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.3]])
    assert top_candidates(m, 3) == [(0, 1, 0.5), (0, 2, 0.5), (0, 3, 0.2), (1, 4, 0.3)]
    assert top_candidates(m, 3, row_offset=10)[0][0] == 10


def test_anchors_do_not_cross():
    x = [W(0, 8), W(10, 18), W(20, 28)]
    t = [W(20, 28), W(0, 8), W(10, 18)]           # the last sentence was moved to the front
    assert find_anchors(BleuScorer.reference(), x, t) == [(0, 1), (1, 2)]


# ---------------------------------------------------------------- second pass

def test_size_rule():
    table = """
        . . . . . . . . .
        . X X X . . . . .
        . X X X . . . . .
        . X X X X X . . .
        . . . X X X X X .
        . . . X X X X X X
        . . . . X X X X X
        . . . . X X X X X
        . . . . . X X X X""".split()          # probed on the original: rows = source gap 0..8, columns = target gap
    for a in range(9):
        for b in range(9):
            assert size_rule(a, b) == (table[a * 9 + b] == "X"), (a, b)


def test_two_to_one_and_one_to_two():
    a, b = W(0, 12, "a"), W(0, 8, "b")
    x, t = [PRE, a, b, POST], [PRE, a + " " + b, POST]
    assert pairs(align(x, t, length_aligner=no_length_aligner)) == [([0], [0]), ([1, 2], [1]), ([3], [2])]
    beads = align(t, x, length_aligner=no_length_aligner)
    assert pairs(beads) == [([0], [0]), ([1], [1, 2]), ([2], [3])]
    assert [bd.kind for bd in beads] == [ANCHOR, HEURISTIC, ANCHOR]


def test_one_neighbour_per_gap():
    # one long sentence against pieces; the largest piece is the anchor and takes one neighbour from each gap
    pieces = [W(0, 6), W(6, 12), W(12, 24), W(24, 30), W(30, 36)]
    beads = align([PRE, W(0, 36), POST], [PRE] + pieces + [POST], length_aligner=no_length_aligner)
    assert pairs(beads)[1] == ([1], [2, 3, 4])


def test_source_side_needs_more_matches_target_side_does_not():
    a, b, junk = W(0, 10, "a"), W(0, 10, "b"), W(0, 6, "j")
    # joining unrelated text only improves the brevity penalty
    assert pairs(align([PRE, a, junk, POST], [PRE, a + " " + b, POST],
                       length_aligner=no_length_aligner))[1] == ([1], [1])
    assert pairs(align([PRE, a + " " + b, POST], [PRE, a, junk, POST],
                       length_aligner=no_length_aligner))[1] == ([1], [1, 2])
    # one more matching word is enough on the source side
    assert pairs(align([PRE, a, junk + " b5", POST], [PRE, a + " " + b, POST],
                       length_aligner=no_length_aligner))[1] == ([1, 2], [1])


def test_one_to_one_heuristic_adds_a_pruned_first_pair():
    g = W(0, 6, "g") + " " + W(0, 6, "u")
    h = W(0, 6, "g") + " " + W(0, 6, "v")
    decoys = [W(0, 6, "g") + " " + W(0, 3, "u") + " " + W(0, 3, f"d{i}x") for i in range(3)]
    junk = [W(0, 12, f"j{i}x") for i in range(3)]
    x = [PRE, g, POST]
    first = align(x, [PRE, h] + junk + [POST] + decoys)          # gap 1 x 4: too uneven for the length aligner
    assert ([1], [1]) in pairs(first)
    assert Bead((1,), (1,), HEURISTIC) in first
    middle = align(x, [PRE, junk[0], h, junk[1], junk[2], POST] + decoys)
    assert all(b.src != (1,) for b in middle)                    # only the first pair of a gap is tried


def test_gap_goes_to_length_aligner_and_kinds():
    x = [PRE, W(0, 8, "a"), W(0, 9, "b"), POST]
    t = [PRE, W(0, 8, "c"), W(0, 9, "d"), POST]
    beads = align(x, t, length_aligner=galechurch_aligner)
    assert pairs(beads) == [([0], [0]), ([1], [1]), ([2], [2]), ([3], [3])]
    assert [b.kind for b in beads] == [ANCHOR, LENGTH, LENGTH, ANCHOR]


def test_no_anchor_at_all_and_empty_input():
    x, t = [W(0, 8, "a"), W(0, 9, "b")], [W(0, 8, "c"), W(0, 9, "d")]
    assert pairs(align(x, t, length_aligner=galechurch_aligner)) == [([0], [0]), ([1], [1])]
    assert align([], []) == [] and align(x, []) == [] and align([], t) == []


def test_beads_are_ordered_and_disjoint_on_random_text():
    rnd = random.Random(11)
    vocab = [f"w{i}" for i in range(60)]
    for _ in range(20):
        t = [" ".join(rnd.choice(vocab) for _ in range(rnd.randint(2, 15))) for _ in range(rnd.randint(1, 40))]
        x = []
        for s in t:                               # the "translation": the target, split, merged and noised
            words = [w if rnd.random() > 0.3 else rnd.choice(vocab) for w in s.split()]
            if rnd.random() < 0.2 and len(words) > 4:
                x += [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
            elif rnd.random() < 0.1:
                continue
            else:
                x.append(" ".join(words))
        beads = align(x, t, length_aligner=galechurch_aligner)
        for side in ("src", "trg"):
            used = [i for b in beads for i in getattr(b, side)]
            assert len(used) == len(set(used))
        for a, b in zip(beads, beads[1:]):
            assert a.src[0] < b.src[0] and a.trg[0] < b.trg[0]


def assert_monotone(beads):
    """Consecutive beads a, b: a ends before b starts on both sides; no sentence in two beads."""
    for side in ("src", "trg"):
        used = [i for b in beads for i in getattr(b, side)]
        assert len(used) == len(set(used)), (side, beads)
    for a, b in zip(beads, beads[1:]):
        assert a.src[-1] < b.src[0] and a.trg[-1] < b.trg[0], (a, b)


def test_gap_preserves_order():
    # from the code review of 15 Nov 2026: anchors 0-0, 1-1, 2-2, 3-4; the bead 2-2 took target 3, and the
    # bead 3-4 then took the target sentence 2 it had left
    x = ['a d f e e b b e c', 'f', 'b c b d a e e c a', 'a e a c a e e c']
    t = ['f f b e', 'd e f c a b', 'd c e', 'd', 'e b f e f c']
    assert_monotone(align(x, t, ChrfScorer(both_directions=True)))


def test_gap_preserves_order_bleu():
    # found by tools/crossing_probe.py (seed 0): anchors 0-0, 4-1; the bead 4-1 takes source 2,3, which
    # frees sentence 4 for the bead 0-0 (4-0 would cross it). The original program gives 0,1-0 and 4-1 here.
    x, t = ['d f', 'd', 'b a', 'c', 'e d d c e'], ['c d d f b', 'b a c e e']
    assert find_anchors(BleuScorer.reference(), x, t) == [(0, 0), (4, 1)]
    beads = align(x, t)
    assert_monotone(beads)
    assert pairs(beads) == [([0], [0]), ([2, 3], [1])]


@pytest.mark.parametrize("scorer", [BleuScorer.reference(), ChrfScorer(both_directions=True)],
                         ids=["bleu", "chrf"])
def test_monotone_on_random_small_problems(scorer):
    # seeded; small vocabularies make the heuristics move beads a lot; empty and repeated sentences; several
    # documents separated by delimiter lines
    rnd = random.Random(2020)
    for it in range(2500):
        vocab = "abcdef"[:rnd.randint(3, 6)]
        pool = []

        def sentence():
            r = rnd.random()
            if r < 0.06:
                return ""
            if r < 0.2 and pool:
                return rnd.choice(pool)
            s = " ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, 9)))
            pool.append(s)
            return s

        n_docs = 1 if rnd.random() < 0.7 else rnd.randint(2, 3)
        x, t = [], []
        for d in range(n_docs):
            if d:
                x.append(".EOA")
                t.append(".EOA")
            x += [sentence() for _ in range(rnd.randint(0 if n_docs > 1 else 1, 8))]
            t += [sentence() for _ in range(rnd.randint(0 if n_docs > 1 else 1, 8))]
        beads = align_documents(x, t, scorer) if n_docs > 1 else align(x, t, scorer)
        try:
            assert_monotone(beads)
        except AssertionError:
            raise AssertionError(f"problem {it}: {x!r} {t!r}") from None


# ---------------------------------------------------------------- length aligner, intersection, delimiters

def test_gale_church_adapter():
    a = galechurch_aligner(["x" * 30, "y" * 10, "z" * 12], ["x" * 31, "y" * 22])
    assert a == [([0], [0]), ([1, 2], [1])]
    assert galechurch_aligner(["abc"], []) == [([0], [])]


def test_intersect_and_flip():
    f = [Bead((0,), (0,), ANCHOR), Bead((1, 2), (1,), HEURISTIC), Bead((3,), (2,), LENGTH)]
    r = [Bead((0,), (0,), ANCHOR), Bead((1,), (1, 2), HEURISTIC), Bead((2,), (4,), LENGTH)]   # other direction
    assert flip(r)[1] == Bead((1, 2), (1,), HEURISTIC)
    assert intersect(f, flip(r)) == f[:2]
    assert intersect(f) == f and intersect() == []


def test_documents_with_hard_delimiters():
    a, b = W(0, 8, "a"), W(0, 8, "b")
    src = [a, ".EOA", b, a]
    translated = [a, ".eoa", b, a]                # the delimiter line was changed by the translation
    trg = [a, ".EOA", b, a]
    assert [d[0] for d in split_documents(trg)] == [0, 2]
    with pytest.raises(ValueError):
        align_documents(translated, [a, b, a], source=src)
    beads = align_documents(translated, trg, source=src, length_aligner=no_length_aligner)
    assert pairs(beads) == [([0], [0]), ([2], [2]), ([3], [3])]   # the second `a` is not aligned across the delimiter


# ---------------------------------------------------------------- against the original Bleualign

def _recorded_aligner():
    store = json.load(open(os.path.join(HERE, "reference", "galechurch_gaps.json")))

    def aligner(src, trg):
        k = hashlib.sha1(("\n".join(src) + "\n\t\n" + "\n".join(trg)).encode("utf-8")).hexdigest()
        return [tuple(b) for b in store[k]]
    return aligner


@needs_repo
@pytest.mark.parametrize("run", [r for r in RUNS if RUNS[r][3]])
def test_anchors_identical_to_the_original(run):
    scorer = BleuScorer.reference()
    for translated, target, ref in load_run(run, REPO):
        assert find_anchors(scorer, translated, target) == [tuple(a) for a in ref["anchors"]]


# identical beads reached on 14 Nov 2026 with the original's own Gale-Church for the gaps
REACHED = {"test_europarlfull": (804, 813), "test_google": (795, 813), "test_europarllight": (700, 723),
           "dev_europarlfull": (361, 374)}


@needs_repo
@pytest.mark.parametrize("run", list(REACHED))
def test_beads_against_the_original(run):
    scorer, aligner = BleuScorer.reference(), _recorded_aligner()
    same = total = 0
    for translated, target, ref in load_run(run, REPO):
        ours = {(b.src, b.trg) for b in align(translated, target, scorer, length_aligner=aligner)}
        theirs = {(tuple(s), tuple(t)) for s, t in ref["beads"]}
        same, total = same + len(ours & theirs), total + len(theirs)
    assert total == REACHED[run][1]
    assert same >= REACHED[run][0]


@needs_repo
def test_both_directions_against_the_original():
    scorer, aligner = BleuScorer.reference(), _recorded_aligner()
    fwd = [align(x, t, scorer, length_aligner=aligner) for x, t, _ in load_run("test_europarlfull", REPO)]
    rev = [align(x, t, scorer, length_aligner=aligner) for x, t, _ in load_run("test_europarlfull_reverse", REPO)]
    ref = reference("test_europarlfull_both_directions")
    same = total = 0
    for f, r, entry in zip(fwd, rev, ref):
        ours = {(b.src, b.trg) for b in intersect(f, flip(r))}
        theirs = {(tuple(s), tuple(t)) for s, t in entry["beads"]}
        same, total = same + len(ours & theirs), total + len(theirs)
    assert total == 644 and same >= 637


# ---- speed without a change of output -----------------------------------------------------------------

def _random_text(rng, n, vocab=6):
    # few distinct words, so that many scores tie and repeats occur
    return [" ".join(f"w{rng.randrange(vocab)}" for _ in range(rng.randrange(1, 6))) for _ in range(n)]


SCORERS = [lambda: BleuScorer.reference(), lambda: ChrfScorer(both_directions=True)]


@pytest.mark.parametrize("make", SCORERS, ids=["bleu", "chrf"])
def test_blocked_candidates_equal_one_global_sort(make):
    """Blocks in both dimensions give exactly the top 3 per row of the whole matrix, ties to the lower column."""
    from translatealign.search import candidates
    rng = random.Random(13)
    for _ in range(30):
        tr, tg = _random_text(rng, rng.randrange(1, 14)), _random_text(rng, rng.randrange(1, 14))
        scorer = make()
        scores, _ = scorer.score_matrix(tr, tg)
        expected = top_candidates(scores, 3)
        for block, col_block in ((500, 2000), (3, 2), (1, 1), (2, 5), (7, 3)):
            assert candidates(make(), tr, tg, 3, block=block, col_block=col_block) == expected


@pytest.mark.parametrize("make", SCORERS, ids=["bleu", "chrf"])
def test_prepared_targets_give_the_same_matrix(make):
    rng = random.Random(14)
    tr, tg = _random_text(rng, 9), _random_text(rng, 11)
    scorer = make()
    s0, m0 = scorer.score_matrix(tr, tg)
    prepared = scorer.prepare(tg)
    s1, m1 = scorer.score_matrix(tr, prepared)
    assert (s0 == s1).all() and (m0 == m1).all()
    s2, m2 = scorer.score_matrix(tr, prepared.slice(3, 8))
    assert (s0[:, 3:8] == s2).all() and (m0[:, 3:8] == m2).all()
    assert len(prepared.slice(3, 100)) == 8            # a slice past the end stops at the end
    for c, s, m in scorer.score_column_blocks(tr, prepared, 4):
        assert (s0[:, c:c + 4] == s).all() and (m0[:, c:c + 4] == m).all()


@pytest.mark.parametrize("make", SCORERS, ids=["bleu", "chrf"])
def test_clearing_the_cache_changes_nothing(make):
    rng = random.Random(15)
    tr, tg = _random_text(rng, 12), _random_text(rng, 12)
    scorer = make()
    first = align(tr, tg, scorer)               # align() clears the scorer's cache at its end
    assert align(tr, tg, scorer) == first == align(tr, tg, make())
    scorer.score_matrix(tr, tg)
    assert scorer._ng.cache_info().currsize > 0
    scorer.clear_cache()
    assert scorer._ng.cache_info().currsize == 0
