"""The translator layer, tested with fake models (nothing is downloaded). The test with a real model is at the
end and runs only with TRANSLATEALIGN_RUN_MODELS=1."""
import os
import warnings

import pytest

from translatealign import cli
from translatealign.evaluate import read_alignment
from translatealign.translate import (ChatTranslator, Checks, DefinitionError, FileTranslator, Seq2SeqTranslator,
                                      TranslationCache, TranslationWarning, Translator, UnknownLanguage,
                                      definition_from_dict, load_definition, load_translator, make_batches)
from translatealign.translate import _hf, base
from translatealign.translate.chat import clean, prompt_fragments
from translatealign.translate.languages import family_of, language_name, madlad_token, nllb_code, opus_prefix


class Fake(Translator):
    """Translates by upper-casing; `script` maps a sentence to the outputs of attempt 0 and 1."""
    name = "fake"
    batch_size = 4
    max_batch_chars = 200

    def __init__(self, script=None, name="fake"):
        self.script, self.name, self.calls = script or {}, name, []

    def translate_batch(self, sentences, source_language, target_language, attempt=0):
        self.calls.append((list(sentences), attempt))
        return [self.script[s][attempt] if s in self.script else s.upper() for s in sentences]


def quiet(translator, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TranslationWarning)
        return translator.translate(*args, **kwargs)


SENTENCES = [f"sentence number {i} " + "word " * (i % 7) for i in range(23)]


# ---- the interface ------------------------------------------------------------------------------------------

def test_same_number_and_order():
    out = Fake().translate(SENTENCES, "de", "fr")
    assert list(out) == [s.strip().upper() for s in SENTENCES]
    assert out.translated == 23 and out.problems == [] and out.from_cache == 0


def test_empty_lines_and_delimiters_are_not_sent_to_the_model():
    fake = Fake()
    out = fake.translate(["one", "", "  ", ".EOA", "two"], "de", "fr", keep=[".EOA"])
    assert list(out) == ["ONE", "", "", ".EOA", "TWO"] and out.skipped == 3
    assert sorted(s for batch, _ in fake.calls for s in batch) == ["one", "two"]


def test_a_repeated_sentence_is_translated_once():
    fake = Fake()
    out = fake.translate(["a b", "c d", "a b", "a b"], "de", "fr")
    assert list(out) == ["A B", "C D", "A B", "A B"] and out.translated == 2
    assert sum(len(batch) for batch, _ in fake.calls) == 2


def test_empty_input():
    assert list(Fake().translate([], "de", "fr")) == []


# ---- batching -----------------------------------------------------------------------------------------------

def test_batches_by_length():
    texts = ["x" * n for n in (5, 50, 7, 49, 6, 48, 100, 1)]
    batches = make_batches(texts, batch_size=3, max_batch_chars=150)
    assert sorted(t for b in batches for t in b) == sorted(texts)
    flat = [len(t) for b in batches for t in b]
    assert flat == sorted(flat, reverse=True)                       # long to short
    assert all(len(b) <= 3 and len(b) * len(b[0]) <= 150 for b in batches)
    assert batches[0] == ["x" * 100] and [len(t) for t in batches[1]] == [50, 49, 48]


def test_a_sentence_longer_than_the_budget_is_a_batch_of_its_own():
    assert make_batches(["x" * 500, "y"], batch_size=8, max_batch_chars=100) == [["x" * 500], ["y"]]


def test_the_translator_uses_its_batch_settings():
    fake = Fake()
    fake.translate(SENTENCES, "de", "fr")
    assert all(len(batch) <= 4 and len(batch) * len(batch[0]) <= 200 for batch, _ in fake.calls)


# ---- cache and resume ---------------------------------------------------------------------------------------

def test_cache_nothing_is_translated_twice(tmp_path):
    path = tmp_path / "c" / "cache.sqlite"
    with TranslationCache(path) as cache:
        first = Fake().translate(SENTENCES, "de", "fr", cache=cache)
        assert len(cache) == 23
    fake = Fake()
    with TranslationCache(path) as cache:                             # a new process
        second = fake.translate(SENTENCES + ["a new one"], "de", "fr", cache=cache)
    assert list(second)[:-1] == list(first) and second.from_cache == 23 and second.translated == 1
    assert [b for b, _ in fake.calls] == [["a new one"]]


def test_cache_key_is_model_languages_and_text(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        Fake().translate(["a b"], "de", "fr", cache=cache)
        for translator, src, trg in ((Fake(name="other"), "de", "fr"), (Fake(), "de", "en"), (Fake(), "nl", "fr")):
            assert translator.translate(["a b"], src, trg, cache=cache).from_cache == 0
        assert Fake().translate(["a b"], "de", "fr", cache=cache).from_cache == 1
        assert len(cache) == 4


def test_resume_after_interruption(tmp_path):
    class Dies(Fake):
        def translate_batch(self, sentences, *a, **k):
            if len(self.calls) == 3:
                raise KeyboardInterrupt
            return super().translate_batch(sentences, *a, **k)

    path = tmp_path / "cache.sqlite"
    with TranslationCache(path) as cache:
        with pytest.raises(KeyboardInterrupt):
            Dies().translate(SENTENCES, "de", "fr", cache=cache)
    with TranslationCache(path) as cache:
        done = len(cache)
        assert done == 12                                             # three batches of four were finished
        fake = Fake()
        out = fake.translate(SENTENCES, "de", "fr", cache=cache)
    assert out.from_cache == done and out.translated == 23 - done
    assert list(out) == [s.strip().upper() for s in SENTENCES]


def test_failed_sentences_are_remembered_and_not_tried_again(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        out = quiet(Fake({"bad one": ["", ""]}), ["good one", "bad one"], "de", "fr", cache=cache)
        assert list(out) == ["GOOD ONE", ""] and len(cache) == 1
        fake = Fake()
        again = quiet(fake, ["good one", "bad one"], "de", "fr", cache=cache)
        assert list(again) == ["GOOD ONE", ""] and fake.calls == [] and again.known_failures == 1
        assert [(p.lines, p.reason) for p in again.problems] == [((1,), base.EMPTY)]
        # on request it is tried again; a success replaces the failure
        third = Fake().translate(["good one", "bad one"], "de", "fr", cache=cache, retry_failed=True)
        assert list(third) == ["GOOD ONE", "BAD ONE"] and third.translated == 1
        fourth = Fake().translate(["bad one"], "de", "fr", cache=cache)
        assert fourth.from_cache == 1 and fourth.problems == [] and fourth.known_failures == 0


# ---- the checks ---------------------------------------------------------------------------------------------

SOURCE = "Der Gipfel wurde am frühen Morgen erreicht."


@pytest.mark.parametrize("output, reason", [
    ("Le sommet fut atteint tôt le matin.", None),
    ("", base.EMPTY),
    ("   \n ", base.EMPTY),
    ("Le sommet fut atteint.\nTôt le matin.", base.SEVERAL_LINES),
    ("Here is the French translation: Le sommet fut atteint tôt le matin.", base.COMMENTARY),
    ("Sure, here you go: Le sommet fut atteint.", base.COMMENTARY),
    ("Translation: Le sommet fut atteint tôt le matin.", base.COMMENTARY),
    ("Le sommet fut atteint tôt le matin. (Note: 'Gipfel' can also mean croissant.)", base.COMMENTARY),
    ("I cannot translate this text.", base.COMMENTARY),
    ("Le sommet fut atteint.<|im_end|>", base.COMMENTARY),
    ("Translate the following German text into French: Le sommet", base.LEAK),
    ("Le sommet " * 40, base.TOO_LONG),
    ("Le", None),                                   # short, but within the slack of 10 characters
])
def test_checks(output, reason):
    checks = Checks(prompt_fragments=["Translate the following German text into French"])
    assert checks.problem(SOURCE, output) == reason


def test_too_short():
    assert Checks().problem(SOURCE * 4, "Le") == base.TOO_SHORT
    assert Checks(min_ratio=0.5, max_ratio=2, slack=0).problem("x" * 10, "y" * 21) == base.TOO_LONG
    assert Checks(min_ratio=0.5, max_ratio=2, slack=0).problem("x" * 10, "y" * 20) is None


@pytest.mark.parametrize("source, output", [         # real translations that an earlier pattern rejected
    ("„Auðvitað er ég tilbúinn til að efna loforð mitt.", "Of course I am ready to keep my promise."),
    ("Ég spurði eftir bróður þeirra auðvitað.", "Of course, I asked for their brother."),
    ("– Víst er að ég vildi að henni stæði á sama um hann.", "▪ Sure, I wanted her to care about him."),
    ("Allt í lagi, ég kasta.", "Okay, I'm throwing."),
    ("Ég get ekki veitt þér það sem þú sækist eftir.", "I am unable to provide you with what you are looking for."),
    ("Ég get ekki stillt mig um að spyrja.", "I can't help but ask."),
])
def test_ordinary_sentences_are_not_commentary(source, output):
    assert Checks().problem(source, output) is None


@pytest.mark.parametrize("output", [
    "Certainly. Here's the English translation of the provided Icelandic phrase: No problem.",
    "Of course, here is the translation: No problem.",
    "Okay, let me translate that: No problem.",
    "Sure, I can translate that. No problem.",
    "I am unable to provide a translation without the text.",
])
def test_chat_commentary_is_still_caught(output):
    assert Checks().problem("-Já minnsta málið.", output) == base.COMMENTARY


def test_commentary_pattern_that_is_in_the_source_too_does_not_count():
    assert Checks().problem("Note: the hut is closed in winter.", "Note : la cabane est fermée en hiver.") is None
    assert Checks().problem("Sure, here it is.", "Sure, here it is.") is None


def test_retry_once_with_other_settings_then_empty_translation():
    script = {"loops": ["la la " * 60, "fine now"],                  # second attempt is good
              "dead": ["", "Here is the translation: x"],            # both attempts fail
              "lines": ["one\ntwo", "one\ntwo"]}
    fake = Fake(script)
    with pytest.warns(TranslationWarning, match="2 sentence"):
        out = fake.translate(["ok one", "loops", "dead", "ok two", "lines", "dead"], "de", "fr")
    assert list(out) == ["OK ONE", "fine now", "", "OK TWO", "", ""]
    assert out.retried == 3
    assert [(p.lines, p.first_reason, p.reason) for p in out.problems] == [
        ((2, 5), base.EMPTY, base.COMMENTARY), ((4,), base.SEVERAL_LINES, base.SEVERAL_LINES)]
    retries = [batch for batch, attempt in fake.calls if attempt == 1]
    assert sorted(retries) == [["dead"], ["lines"], ["loops"]]        # one at a time, once each
    assert "line 3" in base.describe_problems(out.problems)


def test_wrong_number_of_outputs_fails_the_whole_batch_and_is_retried_singly():
    class Merges(Fake):
        def translate_batch(self, sentences, s, t, attempt=0):
            self.calls.append((list(sentences), attempt))
            return [x.upper() for x in sentences][:-1] if len(sentences) > 1 else [sentences[0].upper()]

    fake = Merges()
    out = fake.translate(["a a", "b b", "c c"], "de", "fr")
    assert list(out) == ["A A", "B B", "C C"] and out.retried == 3 and out.problems == []


def test_the_aligner_runs_with_empty_translations():
    src = [" ".join(f"a{i}x{k}" for k in range(6)) for i in range(6)]
    script = {src[2]: ["", ""]}
    translated = quiet(Fake(script), src, "de", "fr")
    assert translated[2] == "" and len(translated) == 6
    beads = cli.align_files(src, [s.upper() for s in src], [translated], scorer="chrf")
    pairs = [(b.src, b.trg) for b in beads]
    assert all(((i,), (i,)) in pairs for i in (0, 1, 3, 4, 5))


# ---- a ready translation ------------------------------------------------------------------------------------

def test_file_translator(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("un\n\ntrois\n", encoding="utf-8")
    out = FileTranslator(str(p)).translate(["eins", "zwei", "drei"], "de", "fr")
    assert list(out) == ["un", "", "trois"] and [x.lines for x in out.problems] == [(1,)]
    with pytest.raises(ValueError, match="3 lines"):
        FileTranslator(str(p)).translate(["eins", "zwei"])


# ---- model definitions and the chat backend -----------------------------------------------------------------

DEFINITION = """
name = "toy"
backend = "chat"
model = "nobody/toy-model"
licence = "none"

[languages]
is = "Icelandic (íslenska)"

[prompt]
mode = "chat"
system = "You are a translator."
template = "Translate the following {source_language} text into {target_language}:\\n{text}"

[generation]
max_new_tokens = 200
batch_size = 2
stop = ["\\n\\n"]

[retry]
repetition_penalty = 1.2
max_new_tokens = 100

[cleanup]
remove = ["<think>.*?</think>"]
strip_prefixes = ["Translation:", "Þýðing:"]
lines = "first"
strip_quotes = true

[checks]
max_ratio = 3.0
extra_leak_patterns = ["^Hér er þýðingin"]
"""


@pytest.fixture
def definition(tmp_path):
    p = tmp_path / "toy.toml"
    p.write_text(DEFINITION, encoding="utf-8")
    return load_definition(str(p))


def test_definition_file(definition):
    d = definition
    assert (d.name, d.backend, d.model) == ("toy", "chat", "nobody/toy-model")
    assert d.setting("batch_size") == 2 and d.setting("max_new_tokens") == 200
    assert d.setting("max_new_tokens", attempt=1) == 100 and d.setting("stop") == ["\n\n"]
    assert d.generate_kwargs() == {} and d.generate_kwargs(1) == {"repetition_penalty": 1.2}
    assert d.setting("new_tokens_per_input_token") == 3.0             # default


@pytest.mark.parametrize("change, message", [
    ({"backend": "magic"}, "backend"),
    ({"promt": {}}, "unknown key"),
    ({"prompt": {"template": "no placeholder"}}, "{text}"),
    ({"prompt": {"template": "{text}", "tempalte": "x"}}, "unknown key"),
    ({"prompt": {"template": "{text}", "content": {"text": "{text}"}}}, "either"),
    ({"prompt": {"mode": "plain", "content": {"text": "{text}"}}}, "chat"),
    ({"cleanup": {"lines": "all"}}, "lines"),
    ({"load": {"quantization": "2bit"}}, "quantization"),
    ({"generation": {"stop": "\n"}}, "stop"),
    ({"model": None}, "model"),
])
def test_definition_errors(change, message):
    data = {"model": "x/y", "prompt": {"template": "{text}"}}
    data.update(change)
    if data["model"] is None:
        del data["model"]
    with pytest.raises(DefinitionError, match=message):
        definition_from_dict(data)


def test_fingerprint_changes_with_what_changes_the_output(definition):
    def variant(**change):
        data = {"model": "x/y", "prompt": {"template": "{text}"}, "generation": {"batch_size": 4}}
        data.update(change)
        return definition_from_dict(data).fingerprint()

    same = variant()
    assert variant(generation={"batch_size": 64}) == same and variant(licence="MIT") == same
    assert variant(load={"device": "cpu"}) == same
    assert variant(generation={"num_beams": 4}) != same and variant(prompt={"template": "T: {text}"}) != same
    assert variant(load={"quantization": "4bit"}) != same and variant(cleanup={"lines": "first"}) != same


def test_prompts(definition):
    seen = []

    def generate(prompts, attempt):
        seen.extend(prompts)
        return ["x"] * len(prompts)

    t = ChatTranslator(definition, generate=generate)
    t.translate(["Hello {text} {braces}."], "en", "is")
    assert seen == [[{"role": "system", "content": "You are a translator."},
                     {"role": "user", "content": "Translate the following English text into Icelandic "
                                                 "(íslenska):\nHello {text} {braces}."}]]


def test_plain_prompt_and_structured_content():
    plain = definition_from_dict({"model": "x/y", "prompt": {
        "mode": "plain", "template": "{source_code}: {text}\n{target_code}:"}})
    t = ChatTranslator(plain, generate=lambda p, a: p)
    assert t.build_prompts(["Guten Tag."], "de", "fr") == ["de: Guten Tag.\nfr:"]
    structured = definition_from_dict({"model": "x/y", "prompt": {"content": {
        "type": "text", "source_lang_code": "{source_code}", "target_lang_code": "{target_code}",
        "text": "{text}"}}})
    t = ChatTranslator(structured, generate=lambda p, a: p)
    assert t.build_prompts(["Guten Tag."], "de", "xx") == [[{"role": "user", "content": [{
        "type": "text", "source_lang_code": "de", "target_lang_code": "xx", "text": "Guten Tag."}]}]]


def test_language_names():
    assert language_name("de") == "German" and language_name("de-CH") == "German"
    assert language_name("is", {"is": "íslenska"}) == "íslenska"
    with pytest.raises(UnknownLanguage):
        language_name("xx")
    t = ChatTranslator(definition_from_dict({"model": "x/y", "prompt": {
        "template": "Into {target_language}: {text}"}}), generate=lambda p, a: ["x"] * len(p))
    with pytest.raises(UnknownLanguage):
        t.translate(["a"], "de", "xx")
    assert list(t.translate(["a"], "xx", "fr")) == ["x"]              # the source name is not in the prompt


@pytest.mark.parametrize("raw, expected", [
    ("Halló heimur.", "Halló heimur."),
    ("  Translation: Halló heimur.  ", "Halló heimur."),
    ("translation: Þýðing: Halló heimur.", "Halló heimur."),
    ("<think>The user wants\nIcelandic.</think>Halló heimur.", "Halló heimur."),
    ("Halló heimur.\n\nNote: 'heimur' means world.", "Halló heimur."),          # stop string
    ("\n\nHalló heimur.\n\nMore.", "Halló heimur."),                            # leading stop string skipped
    ("Halló heimur.\nAnnað.", "Halló heimur."),                                 # lines = first
    ("\"Halló heimur.\"", "Halló heimur."),
    ("„Halló heimur.“", "Halló heimur."),
    ("\"Halló\" sagði \"hann\"", "\"Halló\" sagði \"hann\""),
])
def test_clean(definition, raw, expected):
    assert clean(raw, "Hello world.", definition.cleanup, definition.setting("stop")) == expected


def test_clean_keeps_quotes_of_a_quoted_source_and_line_rules():
    assert clean("\"Halló.\"", "\"Hello.\"", {"strip_quotes": True}) == "\"Halló.\""
    assert clean("a\n\nb", "x", {"lines": "last"}) == "b"
    assert clean("a\n\nb", "x", {"lines": "join"}) == "a b"
    assert clean("a\nb", "x", {}) == "a\nb"                                     # default: kept, the check fails


def test_chat_translator_cleans_checks_and_retries(definition):
    answers = {
        "Good morning.": ["Translation: \"Góðan daginn.\"", None],
        "It rains.": ["Hér er þýðingin: Það rignir.", "Það rignir."],                 # pattern of the definition
        "Leak.": ["Translate the following English text into Icelandic: Leki.", "<think>hm</think>"],
        "Long.": ["mjög " * 30, "Langt."],                                            # max_ratio = 3
    }
    attempts = []

    def generate(prompts, attempt):
        attempts.append(attempt)
        return [answers[p[-1]["content"].split("\n", 1)[1]][attempt] for p in prompts]

    t = ChatTranslator(definition, generate=generate)
    assert prompt_fragments(definition) == ["Translate the following", "You are a translator."]
    out = quiet(t, list(answers), "en", "is")
    assert list(out) == ["Góðan daginn.", "Það rignir.", "", "Langt."]
    assert [(p.source, p.first_reason, p.reason) for p in out.problems] == [("Leak.", base.LEAK, base.EMPTY)]
    assert attempts == [0, 0, 1, 1, 1]                                # two batches of two, three retries


def test_cache_id_is_the_definition(definition, tmp_path):
    other = definition_from_dict({"model": "nobody/toy-model", "prompt": {"template": "Other prompt: {text}"}})
    a = ChatTranslator(definition, generate=lambda p, at: ["a"] * len(p))
    b = ChatTranslator(other, generate=lambda p, at: ["b"] * len(p))
    with TranslationCache(tmp_path / "c.sqlite") as cache:
        assert list(a.translate(["x y"], "en", "is", cache=cache)) == ["a"]
        assert list(b.translate(["x y"], "en", "is", cache=cache)) == ["b"]
        assert a.translate(["x y"], "en", "is", cache=cache).from_cache == 1


def test_the_definitions_in_the_documentation_are_valid():
    import re
    doc = os.path.join(os.path.dirname(__file__), "..", "docs", "MODELS.md")
    with open(doc, encoding="utf-8") as f:
        blocks = re.findall(r"```toml\n(.*?)```", f.read(), flags=re.DOTALL)
    assert len(blocks) == 3
    from translatealign.translate.definition import tomllib
    for block in blocks:
        d = definition_from_dict(tomllib.loads(block))
        assert d.model and d.fingerprint()


# ---- remote code --------------------------------------------------------------------------------------------

def test_remote_code_only_on_the_explicit_option(tmp_path):
    d = definition_from_dict({"model": "x/y", "trust_remote_code": True, "prompt": {"template": "{text}"}})
    with pytest.raises(_hf.RemoteCodeNotAllowed, match="--trust-remote-code"):
        ChatTranslator(d)
    assert ChatTranslator(d, trust_remote_code=True)._remote is True
    assert ChatTranslator(definition_from_dict({"model": "x/y", "prompt": {"template": "{text}"}}),
                          trust_remote_code=True)._remote is False   # the option alone switches nothing on
    with pytest.raises(_hf.RemoteCodeNotAllowed):
        Seq2SeqTranslator("x/nllb-thing", needs_remote_code=True)


# ---- encoder-decoder models: languages ----------------------------------------------------------------------

NLLB_CODES = ["deu_Latn", "fra_Latn", "isl_Latn", "zho_Hans", "zho_Hant", "arb_Arab", "ace_Arab", "ace_Latn",
              "nob_Latn", "eng_Latn"]


def test_language_codes_per_family():
    assert nllb_code("de", NLLB_CODES) == "deu_Latn" and nllb_code("is", NLLB_CODES) == "isl_Latn"
    assert nllb_code("zh", NLLB_CODES) == "zho_Hans" and nllb_code("ar", NLLB_CODES) == "arb_Arab"
    assert nllb_code("no", NLLB_CODES) == "nob_Latn" and nllb_code("zho_Hant", NLLB_CODES) == "zho_Hant"
    assert nllb_code("de", NLLB_CODES, {"de": "gsw_Latn"}) == "gsw_Latn"
    with pytest.raises(UnknownLanguage, match="several scripts"):
        nllb_code("ace", NLLB_CODES)
    with pytest.raises(UnknownLanguage):
        nllb_code("xx", NLLB_CODES)
    vocab = {"<2fr>", "<2is>", "<2fil>"}
    assert madlad_token("fr", vocab.__contains__) == "<2fr>"
    assert madlad_token("tl", vocab.__contains__, {"tl": "fil"}) == "<2fil>"
    with pytest.raises(UnknownLanguage):
        madlad_token("xx", vocab.__contains__)
    assert opus_prefix("fr", []) == "" and opus_prefix("fr", None) == ""
    assert opus_prefix("fr", [">>fra<<", ">>ita<<"]) == ">>fra<< " and opus_prefix("fr", [">>fr<<"]) == ">>fr<< "
    with pytest.raises(UnknownLanguage):
        opus_prefix("is", [">>fra<<", ">>ita<<"])
    assert family_of("Helsinki-NLP/opus-mt-de-fr") == "opus-mt" and family_of("google/madlad400-3b-mt") == "madlad"
    assert family_of("facebook/nllb-200-distilled-600M") == "nllb" and family_of("x/y") is None


class FakeTokenizer:
    unk_token_id = 0
    supported_language_codes = []

    def __init__(self, tokens):
        self.additional_special_tokens = list(tokens)
        self.src_lang = None

    def convert_tokens_to_ids(self, token):
        return self.additional_special_tokens.index(token) + 1 if token in self.additional_special_tokens else 0


def test_seq2seq_prepare():
    nllb = Seq2SeqTranslator("facebook/nllb-200-distilled-600M")
    tok = FakeTokenizer(NLLB_CODES)
    assert nllb.prepare(tok, ["Guten Tag."], "de", "is") == (["Guten Tag."], {"forced_bos_token_id": 3})
    assert tok.src_lang == "deu_Latn"
    madlad = Seq2SeqTranslator("google/madlad400-3b-mt")
    assert madlad.prepare(FakeTokenizer(["<2fr>"]), ["Guten Tag."], "de", "fr") == (["<2fr> Guten Tag."], {})
    opus = Seq2SeqTranslator("Helsinki-NLP/opus-mt-{src}-{trg}")
    assert opus.model_id("de", "fr") == "Helsinki-NLP/opus-mt-de-fr"
    assert opus.prepare(FakeTokenizer([]), ["Guten Tag."], "de", "fr") == (["Guten Tag."], {})
    assert opus.cache_id("de", "fr") != opus.cache_id("fr", "de")
    with pytest.raises(ValueError, match="family"):
        Seq2SeqTranslator("somebody/some-model")
    assert Seq2SeqTranslator("somebody/some-model", family="nllb").family == "nllb"


def test_load_translator(tmp_path, definition):
    p = tmp_path / "seq.toml"
    p.write_text('backend = "seq2seq"\nmodel = "facebook/nllb-200-distilled-600M"\n[generation]\nnum_beams = 4\n'
                 '[checks]\nmax_ratio = 2.5\n', encoding="utf-8")
    t = load_translator(model_file=str(p))
    assert isinstance(t, Seq2SeqTranslator) and t.name == "seq" and t.checks.max_ratio == 2.5
    assert t.definition.generate_kwargs() == {"num_beams": 4}
    assert t.definition.generate_kwargs(1)["repetition_penalty"] == 1.3          # default retry settings
    assert isinstance(load_translator(model="Helsinki-NLP/opus-mt-de-fr"), Seq2SeqTranslator)
    with pytest.raises(ValueError):
        load_translator()


# ---- the command line ---------------------------------------------------------------------------------------

def test_command_line_with_a_model(tmp_path, monkeypatch, capsys):
    src = [" ".join(f"a{i}x{k}" for k in range(6)) for i in range(4)] + [".EOA"] + \
          [" ".join(f"b{i}x{k}" for k in range(6)) for i in range(3)]
    trg = [s.upper() if s != ".EOA" else s for s in src]
    (tmp_path / "s.txt").write_text("\n".join(src) + "\n", encoding="utf-8")
    (tmp_path / "t.txt").write_text("\n".join(trg) + "\n", encoding="utf-8")
    fakes = []

    def fake_loader(model, model_file, **kwargs):
        fakes.append(Fake({src[1]: ["", ""]}))
        return fakes[-1]

    monkeypatch.setattr("translatealign.translate.load_translator", fake_loader)
    args = [str(tmp_path / "s.txt"), str(tmp_path / "t.txt"), "--model", "fake", "-s", "de", "-t", "fr",
            "-o", str(tmp_path / "out"), "--cache", str(tmp_path / "cache.sqlite"), "--save-translations",
            "--scorer", "chrf"]
    assert cli.main(args) == 0
    err = capsys.readouterr().err
    assert "7 sentences translated" in err and "1 sentence(s) of the source text could not" in err
    saved = cli.read_lines(str(tmp_path / "out.translation"))
    assert len(saved) == len(src) and saved[1] == "" and saved[4] == ".EOA" and saved[0] == src[0].upper()
    beads = read_alignment(str(tmp_path / "out.path"))
    assert all(((i,), (i,)) in beads for i in (0, 2, 3, 5, 6, 7))
    assert cli.main(args) == 0                                         # second run: from the cache
    err = capsys.readouterr().err
    assert "0 sentences translated" in err and "6 from the cache, 1 known from the cache to fail" in err
    assert fakes[1].calls == [] and "1 sentence(s) of the source text could not" in err
    assert cli.main(args + ["--retry-failed"]) == 0
    assert "1 sentences translated" in capsys.readouterr().err
    assert sum(len(b) for b, a in fakes[2].calls if a == 0) == 1       # only the failed one again
    with pytest.raises(SystemExit):
        cli.main(args[:2] + ["-o", str(tmp_path / "o"), "--model", "m"])       # languages missing


# ---- a real model (downloaded): Opus-MT German-French -------------------------------------------------------

@pytest.mark.skipif(not os.environ.get("TRANSLATEALIGN_RUN_MODELS"),
                    reason="set TRANSLATEALIGN_RUN_MODELS=1 to download and run Helsinki-NLP/opus-mt-de-fr")
def test_opus_mt_german_french(tmp_path):
    german = ["Der Gipfel wurde am frühen Morgen erreicht.", "", "Das Wetter war schlecht.", ".EOA",
              "Wir stiegen über den Gletscher ab."]
    translator = load_translator(model="Helsinki-NLP/opus-mt-{src}-{trg}")
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        out = translator.translate(german, "de", "fr", cache=cache, keep=[".EOA"])
        again = translator.translate(german, "de", "fr", cache=cache, keep=[".EOA"])
    translator.close()
    assert len(out) == 5 and out[1] == "" and out[3] == ".EOA" and out.problems == []
    assert "sommet" in out[0].lower() and "glacier" in out[4].lower()
    assert list(again) == list(out) and again.from_cache == 3 and again.translated == 0


# ---- cached translations and the current checks -------------------------------------------------------

def _chat(checks):
    return ChatTranslator(definition_from_dict(dict(model="toy", prompt={"template": "{text}"}, checks=checks)),
                          generate=lambda p, a: ["Note: leaked commentary"] * len(p))


def test_cache_applies_current_checks(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        lax = _chat({"leak_patterns": []}).translate(["hello there"], "en", "fr", cache=cache)
        assert list(lax) == ["Note: leaked commentary"] and lax.problems == []
        strict = _chat({})
        out = quiet(strict, ["hello there"], "en", "fr", cache=cache, retry_failed=True)
        assert out.problems or strict.checks.problem("hello there", out[0]) is None
        assert out.rechecked == 1 and out.from_cache == 0 and list(out) == [""]


def test_cached_success_failing_current_checks_is_translated_again(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        ident = Fake().cache_id("de", "fr")
        cache.put_many(ident, "de", "fr", {"good one": "GOOD ONE", "other": "x" * 500})
        fake = Fake()
        out = fake.translate(["good one", "other"], "de", "fr", cache=cache)
        assert list(out) == ["GOOD ONE", "OTHER"] and out.from_cache == 1 and out.rechecked == 1
        assert fake.calls == [(["other"], 0)]
        assert cache.get_many(ident, "de", "fr", ["other"]) == {"other": "OTHER"}      # the entry was replaced
        again = Fake().translate(["good one", "other"], "de", "fr", cache=cache)
        assert again.from_cache == 2 and again.rechecked == 0 and again.translated == 0


def test_a_new_failure_replaces_the_stale_success(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        ident = Fake().cache_id("de", "fr")
        cache.put_many(ident, "de", "fr", {"other": "x" * 500})
        out = quiet(Fake({"other": ["", ""]}), ["other"], "de", "fr", cache=cache)
        assert out.rechecked == 1 and list(out) == [""]
        assert cache.get_many(ident, "de", "fr", ["other"]) == {}
        third = Fake().translate(["other"], "de", "fr", cache=cache)
        assert third.known_failures == 1 and third.rechecked == 0 and third.translated == 0


def test_failure_recorded_under_other_checks_is_tried_again(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        quiet(Fake({"bad one": ["", ""]}), ["bad one"], "de", "fr", cache=cache)
        same = Fake()
        assert quiet(same, ["bad one"], "de", "fr", cache=cache).known_failures == 1 and same.calls == []
        other = Fake()
        other.checks = Checks(slack=11)
        out = quiet(other, ["bad one"], "de", "fr", cache=cache)
        assert out.known_failures == 0 and out.translated == 1 and list(out) == ["BAD ONE"]


def test_checks_fingerprint_follows_every_field():
    base_fp = Checks().fingerprint()
    assert Checks().fingerprint() == base_fp
    for changed in (Checks(min_ratio=0.2), Checks(max_ratio=5), Checks(slack=3), Checks(leak_patterns=["x"]),
                    Checks(prompt_fragments=["x"])):
        assert changed.fingerprint() != base_fp


def test_cache_in_the_old_layout_is_opened_and_its_failures_are_retried(tmp_path):
    import sqlite3
    path = tmp_path / "old.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE translations (key TEXT PRIMARY KEY, model TEXT, source_language TEXT, "
               "target_language TEXT, source TEXT, translation TEXT, created REAL)")
    db.execute("CREATE TABLE failures (key TEXT PRIMARY KEY, model TEXT, source_language TEXT, "
               "target_language TEXT, source TEXT, reason TEXT, first_reason TEXT, output TEXT, created REAL)")
    from translatealign.translate.cache import _key
    key = _key("fake", "de", "fr", "bad one")
    db.execute("INSERT INTO failures VALUES (?, 'fake', 'de', 'fr', 'bad one', 'empty output', 'empty output', '', 0)",
               (key,))
    db.commit()
    db.close()
    with TranslationCache(path) as cache:
        assert cache.get_failures("fake", "de", "fr", ["bad one"]) == {"bad one": ("empty output", "empty output", "")}
        assert cache.get_failures("fake", "de", "fr", ["bad one"], checks=Checks().fingerprint()) == {}
        out = Fake().translate(["bad one"], "de", "fr", cache=cache)       # unknown fingerprint: tried again
        assert out.known_failures == 0 and list(out) == ["BAD ONE"]
    with TranslationCache(path) as cache:                                   # opening twice is harmless
        assert len(cache) == 1


# ---- revision of the weights -----------------------------------------------------------------------

class _FakeModel:
    config = type("C", (), {"max_position_embeddings": None})()
    device = "cpu"

    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_hf(monkeypatch, calls):
    """Replace torch and transformers by objects that record the arguments of from_pretrained."""
    def loader(kind):
        class Auto:
            @staticmethod
            def from_pretrained(model_id, **kwargs):
                calls.append((kind, model_id, kwargs))
                return _FakeModel() if kind != "tokenizer" else type("T", (), {"pad_token_id": 0, "eos_token": "x",
                                                                                "pad_token": "x"})()
        return Auto

    transformers = type("Tf", (), {"AutoTokenizer": loader("tokenizer"), "AutoModelForSeq2SeqLM": loader("model"),
                                   "AutoModelForCausalLM": loader("model")})
    torch = type("Torch", (), {})
    monkeypatch.setattr(_hf, "imports", lambda: (torch, transformers))


def test_revision_is_accepted_and_changes_the_cache_key():
    def make(**load):
        return definition_from_dict({"model": "x/y", "prompt": {"template": "{text}"}, "load": load})

    assert make(revision="abc123").load["revision"] == "abc123"
    assert make(revision="abc123").fingerprint() != make().fingerprint()
    assert make(revision="abc123").fingerprint() != make(revision="def456").fingerprint()
    assert make(revision="abc123").fingerprint() == make(revision="abc123", device="cpu").fingerprint()
    # the cache id of both backends contains the fingerprint, so changed weights do not find old translations
    s2s = Seq2SeqTranslator.from_definition(definition_from_dict(
        {"model": "Helsinki-NLP/opus-mt-de-fr", "backend": "seq2seq", "load": {"revision": "abc123"}}))
    s2s_old = Seq2SeqTranslator.from_definition(definition_from_dict(
        {"model": "Helsinki-NLP/opus-mt-de-fr", "backend": "seq2seq"}))
    assert s2s.cache_id("de", "fr") != s2s_old.cache_id("de", "fr")
    chat = ChatTranslator(make(revision="abc123"), generate=lambda p, a: p)
    assert chat.cache_id("de", "fr") != ChatTranslator(make(), generate=lambda p, a: p).cache_id("de", "fr")


def test_revision_reaches_from_pretrained_in_both_backends(monkeypatch):
    calls = []
    _fake_hf(monkeypatch, calls)
    d = definition_from_dict({"backend": "seq2seq", "model": "Helsinki-NLP/opus-mt-de-fr",
                              "load": {"revision": "abc123"}})
    Seq2SeqTranslator.from_definition(d, device="cpu")._load("de", "fr")
    assert [(k, kw.get("revision")) for k, _, kw in calls] == [("tokenizer", "abc123"), ("model", "abc123")]
    calls.clear()
    d = definition_from_dict({"model": "x/y", "prompt": {"template": "{text}"}, "load": {"revision": "v1.2"}})
    ChatTranslator(d, device="cpu")._load()
    assert [(k, kw.get("revision")) for k, _, kw in calls] == [("tokenizer", "v1.2"), ("model", "v1.2")]
    calls.clear()                                       # without a revision nothing is passed
    d = definition_from_dict({"model": "x/y", "prompt": {"template": "{text}"}})
    ChatTranslator(d, device="cpu")._load()
    assert all("revision" not in kw for _, _, kw in calls)


# ---- generation that stopped at max_new_tokens ------------------------------------------------------

class Cuts(Fake):
    """Reports the sentences in `cut` (by text) as stopped at the token limit."""
    def __init__(self, cut, **kw):
        super().__init__(**kw)
        self.cut = set(cut)

    def translate_batch(self, sentences, source_language, target_language, attempt=0):
        out = super().translate_batch(sentences, source_language, target_language, attempt)
        self.last_truncated = [i for i, s in enumerate(sentences) if s in self.cut]
        return out


def test_truncated_translation_is_warned_about_counted_and_not_cached(tmp_path):
    with TranslationCache(tmp_path / "cache.sqlite") as cache:
        t = Cuts({"long one"})
        with pytest.warns(TranslationWarning, match="stopped at the limit.*first at line 2"):
            out = t.translate(["short", "long one", "fine", "long one"], "de", "fr", cache=cache)
        assert list(out) == ["SHORT", "LONG ONE", "FINE", "LONG ONE"]       # the text is kept
        assert out.truncated == [(2, "long one")] and out.translated == 3
        ident = t.cache_id("de", "fr")
        assert sorted(cache.get_many(ident, "de", "fr", ["short", "long one", "fine"])) == ["fine", "short"]
        again = Cuts({"long one"})
        with pytest.warns(TranslationWarning, match="stopped at the limit"):
            out2 = again.translate(["short", "long one"], "de", "fr", cache=cache)
        assert out2.from_cache == 1 and out2.truncated == [(2, "long one")] and again.calls == [(["long one"], 0)]


def test_no_warning_when_nothing_was_cut():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = Fake().translate(["a b", "c d"], "de", "fr")
    assert out.truncated == []


def test_hit_limit_finds_sequences_without_an_end_token():
    finish = {2, 0}                                   # eos = 2, padding = 0
    rows = [[5, 6, 2, 0, 0], [5, 6, 7, 8, 9], [5, 2, 0, 0, 0], [5, 6, 7, 8, 2]]
    assert _hf.hit_limit(rows, finish) == [1]
    # a sequence that ended at a stop string has no end token, but it did not hit the limit
    assert _hf.hit_limit(rows, finish, ["a", "b\n\nc", "d", "e"], ["\n\n"]) == []      # row 1 ended at a stop string
    assert _hf.hit_limit([[5, 6]], finish, ["x\n\n"], ["\n\n"]) == []
    assert _hf.hit_limit([[5, 6]], finish, ["x"], ["\n\n"]) == [0]


def test_chat_translator_passes_the_backends_finding_on():
    d = definition_from_dict({"model": "x/y", "prompt": {"template": "{text}"}})
    t = ChatTranslator(d, generate=lambda p, a: None)

    def generate(prompts, attempt):
        t.last_truncated = [1]
        return ["one", "two"]

    t._generate = generate
    with pytest.warns(TranslationWarning, match="max_new_tokens"):
        out = t.translate(["aa", "bb"], "en", "fr")
    assert out.truncated == [(2, "bb")]


class _Rows(list):
    shape = (2, 4)

    def tolist(self):
        return [list(r) for r in self]


def test_seq2seq_backend_reports_rows_without_end_token(monkeypatch):
    class Tok:
        pad_token_id, eos_token_id, unk_token_id, model_max_length = 0, 2, 3, 512
        supported_language_codes = None

        def __call__(self, texts, **kw):
            enc = {"input_ids": type("I", (), {"shape": (len(texts), 3)})()}
            return type("E", (dict,), {"to": lambda self, d: self})(enc)

        def batch_decode(self, out, skip_special_tokens=True):
            return ["x"] * len(out)

    class Model(_FakeModel):
        generation_config = type("G", (), {"eos_token_id": 2, "pad_token_id": 0})()

        def generate(self, **kw):
            return _Rows([[3, 7, 8, 2], [3, 7, 8, 9]])        # the second one never ended

    class Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    torch = type("Torch", (), {"inference_mode": staticmethod(lambda: Ctx())})
    monkeypatch.setattr(_hf, "imports", lambda: (torch, None))
    t = Seq2SeqTranslator("Helsinki-NLP/opus-mt-de-fr", device="cpu")
    t._loaded = ("Helsinki-NLP/opus-mt-de-fr", Tok(), Model())
    assert t.translate_batch(["a", "b"], "de", "fr") == ["x", "x"]
    assert t.last_truncated == [1]
