"""tools/translate_sets.py reuses saved translations only if they belong to this source text and this model
."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import translate_sets  # noqa: E402
from evalsets import EvalSet  # noqa: E402
from translatealign.translate import Translator  # noqa: E402


class Upper(Translator):
    def __init__(self, ident="upper#1"):
        self.ident, self.calls = ident, 0

    def cache_id(self, source_language, target_language):
        return self.ident

    def translate_batch(self, sentences, source_language, target_language, attempt=0):
        self.calls += 1
        return [s.upper() for s in sentences]


SRC = ["eins zwei", "drei vier", ".EOA", "fünf sechs"]
TRG = ["one two", "three four", ".EOA", "five six"]


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(translate_sets, "WORK", str(tmp_path))
    state = {"src": SRC, "translator": Upper()}
    monkeypatch.setattr(translate_sets, "load", lambda name: EvalSet(name, "de", "fr", state["src"], TRG, [], []))
    monkeypatch.setattr(translate_sets, "load_translator", lambda model: state["translator"])

    def go(*extra):
        monkeypatch.setattr(sys, "argv", ["translate_sets.py", "toy", "--sets", "s", "--directions", "fwd", *extra])
        translate_sets.main()
        return state

    go.state, go.path = state, str(tmp_path / "s" / "toy.fwd")
    return go


def test_a_second_run_reuses_the_files(run, capsys):
    run()
    t = run.state["translator"]
    first_calls = t.calls
    capsys.readouterr()
    run()
    assert t.calls == first_calls and "have " in capsys.readouterr().out
    meta = json.load(open(run.path + ".json"))
    assert {"source_sha256", "translation_sha256", "cache_id"} <= set(meta) and meta["cache_id"] == "upper#1"


def test_changed_source_text_is_translated_again(run, capsys):
    run()
    run.state["src"] = ["eins zwei", "drei VIERZEHN", ".EOA", "fünf sechs"]
    capsys.readouterr()
    run()
    assert "the source text changed" in capsys.readouterr().out
    assert "VIERZEHN" in open(run.path + ".txt", encoding="utf-8").read()


def test_changed_model_definition_is_translated_again(run, capsys):
    run()
    run.state["translator"] = Upper("upper#2")
    capsys.readouterr()
    run()
    assert "the model or its definition changed" in capsys.readouterr().out
    assert json.load(open(run.path + ".json"))["cache_id"] == "upper#2"


def test_edited_translation_file_is_not_trusted(run, capsys):
    run()
    with open(run.path + ".txt", "a", encoding="utf-8") as f:
        f.write("extra\n")
    capsys.readouterr()
    run()
    assert "changed or damaged" in capsys.readouterr().out


def test_missing_text_file_and_old_metadata_are_not_reusable(run, capsys):
    run()
    os.remove(run.path + ".txt")
    ok, why = translate_sets.reusable(run.path, SRC, "upper#1")
    assert not ok and why == "no files"
    run()                                              # written again
    meta = json.load(open(run.path + ".json"))
    for k in ("source_sha256", "cache_id", "translation_sha256"):
        del meta[k]                                    # a file from before the hashes
    json.dump(meta, open(run.path + ".json", "w"))
    ok, why = translate_sets.reusable(run.path, SRC, "upper#1")
    assert not ok and "before the hashes" in why
