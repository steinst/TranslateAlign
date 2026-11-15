import os
import runpy

import pytest

from reference_data import reference
from translatealign import cli
from translatealign.evaluate import read_alignment, score
from translatealign.search import Bead

REPO = os.environ.get("TRANSLATEALIGN_BLEUALIGN_REPO")
needs_repo = pytest.mark.skipif(not REPO, reason="set TRANSLATEALIGN_BLEUALIGN_REPO to a clone of Bleualign")


def sent(prefix, i, n=6):
    return " ".join(f"{prefix}{i}x{k}" for k in range(n))


# Two documents. Document 1: source sentence 2 has no counterpart. Document 2: one-to-one.
SRC = [sent("a", i) for i in range(5)] + [".EOA"] + [sent("b", i) for i in range(3)]
TRG = [sent("a", i) for i in (0, 1, 3, 4)] + [".EOA"] + [sent("b", i) for i in range(3)]
TRANSLATION = [s if s != ".EOA" else "the delimiter , translated" for s in SRC]
EXPECTED = [((0,), (0,)), ((1,), (1,)), ((3,), (2,)), ((4,), (3,)), ((6,), (5,)), ((7,), (6,)), ((8,), (7,))]


def write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


@pytest.fixture
def files(tmp_path):
    return dict(src=write(tmp_path, "s.txt", SRC), trg=write(tmp_path, "t.txt", TRG),
                tr=write(tmp_path, "s.tr", TRANSLATION), tr_t=write(tmp_path, "t.tr", TRG),
                out=str(tmp_path / "out"))


@pytest.mark.parametrize("scorer", ["bleu", "chrf"])
def test_end_to_end(files, scorer, capsys):
    assert cli.main([files["src"], files["trg"], "--translation", files["tr"], "-o", files["out"],
                     "--scorer", scorer]) == 0
    assert read_alignment(files["out"] + ".path") == EXPECTED
    s, t = cli.read_lines(files["out"] + ".src"), cli.read_lines(files["out"] + ".trg")
    assert s == [SRC[b[0][0]] for b in EXPECTED] and t == [TRG[b[1][0]] for b in EXPECTED]
    err = capsys.readouterr().err
    assert "2 document(s), 8 / 7 sentences" in err and "7 beads" in err and "1 source, 0 target" in err


def test_unaligned_sentences_are_written(files):
    assert cli.main([files["src"], files["trg"], "--translation", files["tr"], "-o", files["out"],
                     "--unaligned", "-q"]) == 0
    beads = read_alignment(files["out"] + ".path")
    assert beads == EXPECTED[:2] + [((2,), ())] + EXPECTED[2:]
    assert cli.read_lines(files["out"] + ".src")[2] == SRC[2] and cli.read_lines(files["out"] + ".trg")[2] == ""
    # every sentence exactly once, no delimiter line
    assert sorted(i for b in beads for i in b[0]) == [0, 1, 2, 3, 4, 6, 7, 8]
    assert sorted(j for b in beads for j in b[1]) == [0, 1, 2, 3, 5, 6, 7]


def test_both_directions_and_gold(files, tmp_path, capsys):
    g1 = write(tmp_path, "g1", ["[0]:[0]", "[1]:[1]", "[2]:[]", "[3]:[2]", "[4]:[3]"])
    g2 = write(tmp_path, "g2", ["[0]:[0]", "[1, 2]:[1, 2]"])
    assert cli.main([files["src"], files["trg"], "--translation", files["tr"], "--translation-of-target",
                     files["tr_t"], "-o", files["out"], "--gold", g1, g2, "-q"]) == 0
    assert read_alignment(files["out"] + ".path") == EXPECTED
    out = capsys.readouterr().out
    assert "5 of 7 beads identical to a gold bead; 6 gold beads" in out
    assert "precision    0.714   1.000" in out


def test_errors(files, tmp_path, capsys):
    short = write(tmp_path, "short.tr", TRANSLATION[:-1])
    assert cli.main([files["src"], files["trg"], "--translation", short, "-o", files["out"]]) == 2
    assert "8 lines" in capsys.readouterr().err
    assert cli.main([files["src"], files["trg"], "--translation", str(tmp_path / "missing"), "-o", files["out"]]) == 2
    one_doc = write(tmp_path, "one.txt", [l for l in TRG if l != ".EOA"])
    assert cli.main([files["src"], one_doc, "--translation", files["tr"], "-o", files["out"]]) == 2
    assert "documents" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main([files["src"], files["trg"], "-o", files["out"]])


def test_with_unaligned_and_per_document():
    src, trg = ["a", "b", ".EOA", "c", "d"], ["x", ".EOA", "y", "z", "w"]
    beads = [Bead((1,), (0,), "anchor"), Bead((3, 4), (3,), "heuristic")]
    full = cli.with_unaligned(beads, src, trg)
    assert full == [((0,), ()), ((1,), (0,)), ((), (2,)), ((3, 4), (3,)), ((), (4,))]
    assert cli.per_document(full, src, trg) == [[((0,), ()), ((1,), (0,))],
                                                [((), (0,)), ((0, 1), (1,)), ((), (2,))]]


# ---------------------------------------------------------------- the published table (docs/RESULTS.md section 1)

def gold_documents(name):
    ev = os.path.join(REPO, "eval")
    if name == "eval1989":
        g = runpy.run_path(os.path.join(ev, "goldeval.py"))
        return [g["gold"][g["gold1990map"][d]] for d in range(7)]
    return [runpy.run_path(os.path.join(ev, "golddev.py"))["goldalign"]]


# reference file, set, strict F1 and lax F1 of the original program (its own report)
TABLE = [("test_europarlfull", "eval1989", 0.807, 0.948), ("test_google", "eval1989", 0.809, 0.950),
         ("test_europarllight", "eval1989", 0.660, 0.832),
         ("test_europarlfull_both_directions", "eval1989", 0.786, 0.842), ("dev_europarlfull", "eval1957", 0.742, 0.957)]


@needs_repo
@pytest.mark.parametrize("ref_name,name,f_strict,f_lax", TABLE)
def test_scoring_reproduces_the_original_report(ref_name, name, f_strict, f_lax):
    """Our scoring of the original's beads gives the numbers the original printed."""
    r = score(gold_documents(name), [d["beads"] for d in reference(ref_name)])
    assert round(r["f1_strict"], 3) == f_strict and round(r["f1_lax"], 3) == f_lax


# strict F1 through the command line (BLEU, standard Gale-Church), as in docs/RESULTS.md
OURS = {"test_europarlfull": 0.799, "test_google": 0.806, "test_europarllight": 0.696,
        "test_europarlfull_both_directions": 0.781, "dev_europarlfull": 0.729}


@needs_repo
@pytest.mark.parametrize("ref_name,name,f_strict,f_lax", TABLE)
def test_command_line_on_the_evaluation_set(ref_name, name, f_strict, f_lax, tmp_path):
    ev = os.path.join(REPO, "eval")
    tr = ref_name.split("_")[1]
    argv = [f"{ev}/{name}.de", f"{ev}/{name}.fr", "--translation", f"{ev}/{name}.{tr}.fr",
            "-o", str(tmp_path / "o"), "-q", "--scorer", "bleu"]
    if ref_name.endswith("both_directions"):
        argv += ["--translation-of-target", f"{ev}/{name}.{tr}.de"]
    assert cli.main(argv) == 0
    docs = cli.per_document(read_alignment(str(tmp_path / "o.path")), cli.read_lines(argv[0]), cli.read_lines(argv[1]))
    r = score(gold_documents(name), docs)
    assert round(r["f1_strict"], 3) == OURS[ref_name]
    # within 0.015 of the original, except with the weakest translation, where the standard Gale-Church is better
    assert abs(r["f1_strict"] - f_strict) < 0.015 or (tr == "europarllight" and r["f1_strict"] > f_strict)
    assert abs(r["f1_lax"] - f_lax) < 0.025


# ---- outputs never overwrite inputs --------------------------------------------------------------

def run_cli(*argv):
    return cli.main([*map(str, argv), "-q"])


def test_output_does_not_destroy_input(tmp_path):
    s, t, tr = tmp_path / "book.src", tmp_path / "book.trg", tmp_path / "tr"
    original = "hello world\nlost sentence\n"
    s.write_text(original)
    t.write_text("hello world\n")
    tr.write_text("hello world\nzzzzzzzzzzzzzz\n")
    assert run_cli(s, t, "--translation", tr, "-o", tmp_path / "book") == 2
    assert s.read_text() == original
    assert t.read_text() == "hello world\n"


@pytest.mark.parametrize("which", ["translation", "translation-of-target", "gold"])
def test_output_collides_with_other_inputs(files, tmp_path, capsys, which):
    out = files["out"]
    victim = out + ".path"
    open(victim, "w").write("0:0\n")
    args = [files["src"], files["trg"], "--translation", files["tr"], "-o", out]
    if which == "translation-of-target":
        args += ["--translation-of-target", victim]
    elif which == "gold":
        args += ["--gold", victim]
    else:
        args[3] = victim + ""  # --translation = OUT.path
    assert run_cli(*args) == 2
    err = capsys.readouterr().err
    assert victim in err
    assert open(victim).read() == "0:0\n"


def test_save_translations_collides_with_translation(files, capsys):
    # --save-translations would write OUT.translation: here that is the --translation file
    out = files["out"]
    os.replace(files["tr"], out + ".translation")
    with open(out + ".translation") as f:
        before = f.read()
    assert run_cli(files["src"], files["trg"], "--translation", out + ".translation", "-o", out,
                   "--save-translations") == 2
    assert out + ".translation" in capsys.readouterr().err
    with open(out + ".translation") as f:
        assert f.read() == before


def test_save_translations_collides_with_target_translation(files, capsys):
    out = files["out"]
    os.replace(files["tr_t"], out + ".translation-of-target")
    assert run_cli(files["src"], files["trg"], "--translation", files["tr"], "--translation-of-target",
                   out + ".translation-of-target", "-o", out, "--save-translations") == 2


def test_output_symlink_to_input(files, tmp_path, capsys):
    link = tmp_path / "link.src"
    try:
        os.symlink(files["src"], link)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    with open(files["src"]) as f:
        before = f.read()
    # OUT.src is a symlink to the source file
    os.replace(link, files["out"] + ".src")
    assert run_cli(files["src"], files["trg"], "--translation", files["tr"], "-o", files["out"]) == 2
    with open(files["src"]) as f:
        assert f.read() == before


def test_output_hard_link_to_input(files, capsys):
    try:
        os.link(files["trg"], files["out"] + ".trg")
    except (OSError, NotImplementedError):
        pytest.skip("no hard links here")
    with open(files["trg"]) as f:
        before = f.read()
    assert run_cli(files["src"], files["trg"], "--translation", files["tr"], "-o", files["out"]) == 2
    with open(files["trg"]) as f:
        assert f.read() == before


def test_output_through_other_spelling_of_path(files, tmp_path):
    # ./../dir/out.src names the same file as the input
    src = tmp_path / "book.src"
    os.replace(files["src"], src)
    spelled = os.path.join(str(tmp_path), "sub", "..", "book")
    os.makedirs(tmp_path / "sub", exist_ok=True)
    assert run_cli(src, files["trg"], "--translation", files["tr"], "-o", spelled) == 2


def test_existing_output_survives_failing_write(files, monkeypatch):
    out = files["out"]
    for suffix in (".path", ".src", ".trg"):
        open(out + suffix, "w").write("old\n")
    real = cli.format_bead
    calls = []

    def failing(s, t):
        calls.append(1)
        if len(calls) == 3:
            raise OSError("disk full")
        return real(s, t)

    monkeypatch.setattr(cli, "format_bead", failing)
    with pytest.raises(OSError):
        run_cli(files["src"], files["trg"], "--translation", files["tr"], "-o", out)
    for suffix in (".path", ".src", ".trg"):
        assert open(out + suffix).read() == "old\n"
    leftovers = [p for p in os.listdir(os.path.dirname(out)) if p.endswith(".tmp") or p.startswith("tmp")]
    assert leftovers == []


def test_no_translation_wasted_before_refusal(files, monkeypatch):
    # the check runs before any model is loaded
    def boom(*a, **k):
        raise AssertionError("translation started")
    monkeypatch.setattr(cli, "translate_files", boom)
    open(files["out"] + ".src", "w").close()
    assert run_cli(files["out"] + ".src", files["trg"], "--model", "opus-mt", "-s", "de", "-t", "fr",
                   "-o", files["out"]) == 2


def test_check_outputs_unit(tmp_path):
    a = tmp_path / "a"
    a.write_text("x")
    assert cli.find_collision([str(tmp_path / "b")], [str(a)]) is None
    assert cli.find_collision([str(a)], [str(tmp_path / "." / "a")]) == (str(a), str(tmp_path / "." / "a"))
