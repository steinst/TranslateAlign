import pytest

from translatealign.evaluate import counts, format_bead, read_alignment, score

GOLD = [((0,), (0, 1)), ((1,), (2,)), ((2, 3), (3,)), ((4,), ()), ((), (4,)), ((5,), (5,))]


def test_identical_alignment_scores_one():
    r = score([GOLD], [GOLD])
    assert r["gold"] == r["test"] == r["strict"] == 4          # beads with an empty side do not count
    assert all(r[k] == 1.0 for k in ("precision_strict", "recall_strict", "f1_strict",
                                     "precision_lax", "recall_lax", "f1_lax"))


def test_strict_and_lax_by_hand():
    test = [((0,), (0,)),            # overlaps gold ([0], [0, 1]): lax only
            ((1,), (2,)),            # identical
            ((2,), (3,)), ((3,), (4,)),   # first overlaps gold ([2, 3], [3]); second overlaps on one side only
            ((4,), (9,)),            # source sentence that is unaligned in gold: wrong
            ((6,), ())]              # empty side: ignored
    c = counts(GOLD, test)
    assert c == dict(test=5, gold=4, strict=1, lax_test=3, lax_gold=3)
    r = score([GOLD], [test])
    assert r["precision_strict"] == pytest.approx(1 / 5) and r["recall_strict"] == pytest.approx(1 / 4)
    assert r["precision_lax"] == pytest.approx(3 / 5) and r["recall_lax"] == pytest.approx(3 / 4)
    assert r["f1_strict"] == pytest.approx(2 * 0.2 * 0.25 / 0.45)


def test_documents_are_summed_not_averaged():
    a = score([GOLD, [((0,), (0,))]], [GOLD, [((0,), (1,))]])
    assert a["precision_strict"] == pytest.approx(4 / 5) and a["recall_strict"] == pytest.approx(4 / 5)
    with pytest.raises(ValueError):
        score([GOLD], [GOLD, GOLD])


def test_empty_input():
    r = score([[]], [[]])
    assert r["f1_strict"] == 0.0 and r["f1_lax"] == 0.0


def test_read_alignment(tmp_path):
    p = tmp_path / "a.path"
    p.write_text("[0, 1]:[2]\n[]:[3]\n[2]:[4]:0.83\n\n", encoding="utf-8")
    assert read_alignment(str(p)) == [((0, 1), (2,)), ((), (3,)), ((2,), (4,))]
    assert format_bead((0, 1), (2,)) == "[0, 1]:[2]" and format_bead((), (3,)) == "[]:[3]"
    p.write_text("[0]:[0]\nnonsense\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_alignment(str(p))
