import pytest

import translatealign
from translatealign import citations, cli
from translatealign.translate import ModelDefinition

from test_cli import EXPECTED, SRC, TRANSLATION, TRG, write


def test_align_gives_pairs_with_text_and_kind():
    pairs = translatealign.align(SRC, TRG, translation=TRANSLATION)
    assert [(p.src, p.trg) for p in pairs] == EXPECTED
    assert pairs[0].source == SRC[0] and pairs[0].target == TRG[0]
    assert {p.kind for p in pairs} <= {"anchor", "heuristic", "length"}


def test_align_equals_the_command_line(tmp_path):
    files = {n: write(tmp_path, n, lines) for n, lines in (("s", SRC), ("t", TRG), ("tr", TRANSLATION))}
    out = str(tmp_path / "out")
    assert cli.main([files["s"], files["t"], "--translation", files["tr"], "-o", out, "-q", "--unaligned"]) == 0
    pairs = translatealign.align(SRC, TRG, translation=TRANSLATION, unaligned=True)
    with open(out + ".path", encoding="utf-8") as f:
        assert len(f.read().splitlines()) == len(pairs)
    assert [p for p in pairs if not p.trg] == [translatealign.Pair((2,), (), SRC[2], "", None)]


def test_several_translations_and_the_other_direction():
    one = translatealign.align(SRC, TRG, translation=[TRANSLATION, TRANSLATION])
    assert [(p.src, p.trg) for p in one] == EXPECTED
    back = [s if s != ".EOA" else "x" for s in TRG]
    both = translatealign.align(SRC, TRG, translation=TRANSLATION, translation_of_target=back, scorer="bleu")
    assert [(p.src, p.trg) for p in both] == EXPECTED


def test_errors():
    with pytest.raises(TypeError):
        translatealign.align("a text", TRG, translation=TRANSLATION)
    with pytest.raises(ValueError):
        translatealign.align(SRC, TRG)
    with pytest.raises(ValueError):
        translatealign.align(SRC, TRG, translation=TRANSLATION, scorer="comet")
    with pytest.raises(ValueError):
        translatealign.align(SRC, TRG, model="opus-mt")       # no languages


def test_citations_of_a_run():
    keys = [c.key for c in citations.for_run("chrf")]
    assert keys == ["sennrich-volk-2010", "sennrich-volk-2011", "gale-church-1993", "popovic-2015", "popovic-2017"]
    assert "papineni-etal-2002" in [c.key for c in citations.for_run("bleu")]
    text = translatealign.cite(model="opus-mt")
    assert "Tiedemann" in text and "Sennrich" in text and "chrF++" in text
    assert "translation file" in translatealign.cite()
    bib = translatealign.cite(scorer="bleu", model="facebook/nllb-200-distilled-600M", bibtex=True)
    assert bib.count("@") == 5 and "nllb-2022" in bib and bib.count("{") == bib.count("}")


def test_model_citations():
    card = citations.for_model(ModelDefinition(name="x", model="org/some-model", prompt={"template": "{text}"}))
    assert "huggingface.co/org/some-model" in card.text and card.bibtex.startswith("@misc{org-some-model,")
    own = citations.for_model(ModelDefinition(name="x", model="org/m", prompt={"template": "{text}"},
                                              citation="@article{me-2026,\n  title = {My {MT} Model},\n}"))
    assert own.key == "me-2026" and own.text == "My MT Model"
    line = citations.for_model(ModelDefinition(name="x", model="org/m", prompt={"template": "{text}"},
                                               citation="Me (2026). My model."))
    assert line.text == "Me (2026). My model." and line.bibtex == ""
    for name in ("opus-mt", "nllb-200", "madlad-400", "translategemma-4b", "qwen-mt-lumen"):
        assert citations.for_model(citations.model_definition(name)).text


def test_command_line_citations(tmp_path, capsys):
    assert cli.main(["--bibtex", "--scorer", "bleu"]) == 0
    out = capsys.readouterr().out
    assert "@inproceedings{papineni-etal-2002," in out and "sennrich-volk-2011" in out
    files = {n: write(tmp_path, n, lines) for n, lines in (("s", SRC), ("t", TRG), ("tr", TRANSLATION))}
    assert cli.main([files["s"], files["t"], "--translation", files["tr"], "-o", str(tmp_path / "o")]) == 0
    captured = capsys.readouterr()
    assert "Please cite" in captured.err and "Popović (2017)" in captured.err and captured.out == ""
    assert cli.main([files["s"], files["t"], "--translation", files["tr"], "-o", str(tmp_path / "o"), "-q",
                     "--bibtex"]) == 0
    captured = capsys.readouterr()
    assert captured.err == "" and "@inproceedings{popovic-2017," in captured.out
    with pytest.raises(SystemExit):
        cli.main(["--translation", files["tr"], "-o", str(tmp_path / "o")])     # no SRC, TRG
    with pytest.raises(SystemExit):
        cli.main([files["s"], files["t"], "--translation", files["tr"]])        # no -o
