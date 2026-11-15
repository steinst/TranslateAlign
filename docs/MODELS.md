# TranslateAlign – translating with a model

The translator layer, model definition format, and validation checks. Section 6 covers
the shipped definitions, speed, memory, and licences.

## 1. What the layer does

    translator.translate(sentences, source_language, target_language) -> the same number of sentences

for any model (`src/translatealign/translate/`). Around the model call it does the following, the same for
every backend:

* **Lines that need no translation**: empty lines stay empty; lines given as `keep` (the document delimiter
  `.EOA`) come back unchanged.
* **Each distinct sentence once**; repeated sentences share the result.
* **Cache on disk** (one SQLite file; `--cache FILE`, else `$TRANSLATEALIGN_CACHE`, else
  `~/.cache/translatealign/translations.sqlite`; `--no-cache`). The key is the model with all settings that can
  change its output, the two languages and the text. Every batch is committed when it is done: a run that is
  interrupted goes on where it stopped, and nothing is translated twice. A definition file that is edited
  (prompt, generation settings, clean-up, quantisation) gets new keys; batch size, device and the `[checks]`
  do not change the key.
* **Batches by length**: sentences sorted from long to short (an out-of-memory error comes at once, not after
  an hour), at most `batch_size` sentences and `max_batch_chars` characters (sentences x the longest one) per
  batch. The number of new tokens per batch is limited to `new_tokens_per_input_token` x the longest input
  (+ 16), at most `max_new_tokens`.
* **Checks on every output** (section 4), **one retry** of each failing sentence alone with the retry
  settings, and if that fails too: the sentence gets an **empty translation**, is listed with the reason
  (`.problems` of the result; a warning on the command line with the first five), and the aligner runs on; such
  a sentence can only be aligned by the length-based step.
* **Failures are remembered** in the cache, so that the model is not loaded on every run for the same
  hopeless sentences; `--retry-failed` tries them again. A failure is stored with a fingerprint of the
  `[checks]` it failed under (`Checks.fingerprint()`, a hash of all its fields, column `checks` of the table
  `failures`); it counts only under the same checks, so changed checks give the sentence another try. A cache
  file made before version 0.1.1 gets the column added when it is opened; its old failures have no fingerprint
  and are tried again once.
* **Cached translations are checked again** with the current checks (the checks are not part of the cache key,
  so an entry may have been stored under looser ones). One that fails is translated again as if it were not in
  the cache, and the new result replaces the entry (a new failure removes it). The summary line says "n cached
  translations failed the current checks and were translated again"; `.rechecked` of the result.

Command line:

    translatealign SRC TRG --model "Helsinki-NLP/opus-mt-{src}-{trg}" -s de -t fr -o OUT
    translatealign SRC TRG --model-file my_model.toml -s en -t is --translate both -o OUT

`--translate source` (default) translates `SRC` into the language of `TRG`; `target` the other way; `both`
makes two runs and keeps the beads found in both. `--save-translations` writes `OUT.translation` and
`OUT.translation-of-target`. `--device`, `--trust-remote-code`, `--retry-failed`. Needs
`pip install "translatealign[models]"` (PyTorch, transformers, sentencepiece); 4-bit and 8-bit loading needs
`"translatealign[quantized]"` (bitsandbytes, accelerate).

Python:

    from translatealign.translate import load_translator, TranslationCache
    translator = load_translator(model="Helsinki-NLP/opus-mt-{src}-{trg}")     # or model_file="x.toml"
    with TranslationCache() as cache:
        french = translator.translate(german, "de", "fr", cache=cache, keep=[".EOA"])
    french.problems, french.translated, french.from_cache, french.retried

## 2. Backends

| backend | file | for |
|---|---|---|
| `seq2seq` | `seq2seq.py` | encoder-decoder models from Hugging Face: families `opus-mt`, `nllb`, `madlad` |
| `chat` | `chat.py` | decoder models, driven by a model definition file |
| file | `file.py` | a ready translation (`--translation FILE`; `FileTranslator`) |

**Languages.** The user gives ISO 639-1 codes (`de`, `fr`, `is`) or the model's own code. Per family
(`languages.py`):

| family | how the model is told | code used |
|---|---|---|
| `opus-mt` | one model per pair: `{src}` and `{trg}` in the model id are replaced by the codes. A model with several target languages gets `>>fra<< ` in front of the sentence | the code, or its ISO 639-3 form, whichever the tokenizer lists |
| `nllb` | source language on the tokenizer, target language as first decoder token | FLORES-200 (`deu_Latn`): ISO 639-3 + the only script the model has for it; a table for the exceptions (`ar` -> `arb_Arab`, `zh` -> `zho_Hans`, `no` -> `nob_Latn`, ...); several scripts and no table entry -> an error that lists them |
| `madlad` | `<2fr> ` in front of the sentence | the code, or its ISO 639-3 form, whichever is in the vocabulary |

The family is read from the model id (`opus-mt`, `nllb`, `madlad` in it); otherwise it must be given in a
definition file. `[languages]` in a definition file overrides any code. For chat models the codes are turned
into names for the prompt (English names of 79 languages are built in; `[languages]` adds or changes names).

## 3. Model definition files (TOML)

One file describes one model. Unknown keys are an error. Only `model` is required (and a prompt for a chat
model).

```toml
name = "my-model"                 # default: the file name; shown in messages
backend = "chat"                  # "chat" (default) or "seq2seq"
model = "organisation/model-id"   # Hugging Face id or a directory; seq2seq: may contain {src} and {trg}
family = "nllb"                   # seq2seq only: opus-mt, nllb, madlad (default: read from the model id)
licence = "Apache-2.0"            # information only (also: description, citation)
trust_remote_code = false         # true = the model needs code from its repository; it is then loaded
                                  # only with the explicit option --trust-remote-code

[languages]                       # chat: code -> name in the prompt; seq2seq: code -> the model's code
is = "Icelandic"

[prompt]                          # chat only
mode = "chat"                     # "chat": the tokenizer's chat template; "plain": the text as it is
system = "You are a translator."  # optional system message (mode "chat")
template = "Translate the following {source_language} text into {target_language}:\n{text}"
response_prefix = ""              # text put at the start of the answer (after the chat template)
template_args = { enable_thinking = false }   # extra arguments for the chat template

[generation]
batch_size = 16                   # ours (defaults shown)
max_batch_chars = 8000
max_new_tokens = 256
new_tokens_per_input_token = 3.0
max_input_tokens = 0              # seq2seq: 0 = the model's limit; longer sentences are cut
stop = ["\n\n"]                   # the answer ends before the first of these
num_beams = 1                     # every other key goes to the model's generate()

[retry]                           # the second attempt: these override [generation]
repetition_penalty = 1.2
stop = ["\n"]

[cleanup]                         # chat only; applied in this order, after the cut at a stop string
remove = ["<think>.*?</think>"]   # regular expressions (dot matches newline) deleted from the answer
strip_prefixes = ["Translation:"] # removed from the start, case-insensitive
lines = "keep"                    # "keep" (default: more than one line fails the check), "first", "last", "join"
strip_quotes = false              # remove quotation marks round the whole answer if the source has none

[load]
dtype = "auto"                    # float32, float16, bfloat16
quantization = "none"             # "4bit" (NF4), "8bit"; needs bitsandbytes
device = "cuda"                   # default: a GPU if there is one; --device overrides
model_class = "AutoModelForCausalLM"   # chat: another transformers class if the model needs one
revision = "a1b2c3d..."           # a Hugging Face commit, tag or branch; tokenizer and model are loaded at it

[checks]
min_ratio = 0.1667                # length of the translation / length of the source, in characters
max_ratio = 6.0
slack = 10                        # characters added to both bounds (short sentences)
extra_leak_patterns = ["^Hér er þýðingin"]   # added to the built-in commentary patterns
# leak_patterns = [...]           # replaces the built-in ones
```

**Placeholders** in `template`, `system` and `[prompt.content]`: `{text}`, `{source_language}`,
`{target_language}` (names), `{source_code}`, `{target_code}` (the codes as the user gave them). The text is
inserted as it is; braces in it are safe.

**Structured messages.** Some chat templates want fields, not a text. `[prompt.content]` replaces `template`;
the user message then has one content item with these fields. TranslateGemma (used to check
the chat backend from the local Hugging Face cache; shipped definitions are in section 6):

```toml
name = "translategemma-4b"
backend = "chat"
model = "google/translategemma-4b-it"
licence = "Gemma Terms of Use (gated model)"

[prompt]
mode = "chat"

[prompt.content]                  # its chat template writes the prompt from the language codes
type = "text"
source_lang_code = "{source_code}"
target_lang_code = "{target_code}"
text = "{text}"

[generation]
max_new_tokens = 256
batch_size = 16
stop = ["\n\n"]

[retry]
repetition_penalty = 1.2
stop = ["\n"]

[load]
dtype = "bfloat16"

[cleanup]
lines = "join"
```

`revision` pins the weights. It is part of the cache key (the definition's fingerprint), so a model
that was changed under the same name never finds the translations of the old one. Without it the newest
version is used, which the cache cannot tell from an older one: give a `revision` where results must be
reproducible. A local directory has no revision (leave it out).

A seq2seq definition is short:

```toml
backend = "seq2seq"
model = "facebook/nllb-200-distilled-600M"     # CC BY-NC 4.0
[generation]
num_beams = 4
[languages]
de = "deu_Latn"
```

Notes:
* Decoding is greedy (`do_sample = false`) for chat models unless the definition says otherwise, also where
  the model's own configuration samples: the cache then holds one answer per sentence.
* Without a `[retry]` table a chat model's second attempt repeats the first with the same settings (and
  normally fails again); give at least another stop string or a repetition penalty. For seq2seq models the
  default retry is greedy search with `repetition_penalty = 1.3` and `no_repeat_ngram_size = 4`.
* With `<think>` blocks to remove, do not use `"\n"` as a stop string (the cut comes first); use
  `lines = "last"`.
* A stop string at the very start of an answer is skipped (models that open with a newline).

## 4. The checks

Applied to the cleaned output of every sentence (`Checks` in `base.py`):

| check | fails when |
|---|---|
| one line per line | the model returns another number of outputs than inputs (the whole batch fails and is retried sentence by sentence), or an output has more than one line |
| not empty | the output is empty for a non-empty sentence |
| no leaked prompt | a fixed piece of the prompt (15 characters or more of `template` or `system`) is in the output and not in the source; chat-template tokens (`<|im_end|>`, `<start_of_turn>`, `<think>`) |
| no commentary | the output matches a commentary pattern ("Here is the translation", "Sure, here is the translation:", "Translation:", "Note:", "(Note: ...)", "I cannot translate / provide a translation", "As an AI") and the source does not match it. The "Sure / Of course / Okay ..." and "I cannot ..." patterns ask for words about the translation: before, they rejected ordinary sentences such as "Of course I am ready ..." |
| length | characters: outside `min_ratio * source - slack ... max_ratio * source + slack` |

**Generation that hit the limit.** Both backends also notice a sequence that stopped because
`max_new_tokens` was reached (it did not end with an end-of-sequence token, and for a chat model did not end at
a stop string): the translation may be cut off. It is not a failure (a cut translation still aligns), but the
command line warns with the line number, the summary line counts them, `.truncated` of the result lists
`(line, sentence)`, and such a translation is **not stored in the cache**, so that the warning comes back on
every run until `max_new_tokens` is raised (the output may differ then). A backend reports it through
`Translator.last_truncated` (positions in the batch). Input that the seq2seq tokenizer cut at `max_input_tokens`
is a different limit and is still not reported.

What they do **not** catch: commentary in other languages than English (add `extra_leak_patterns`); the
prompt of a model whose chat template writes the prompt itself (TranslateGemma: nothing of it is known to the
program); a wrong or partly missing translation of plausible length; a `lines = "first"` rule that drops a
real second line. The default length bounds (1/6 to 6) are wide because of pairs with different scripts;
for European pairs `max_ratio = 3` is enough.

## 5. What was run (14 Nov 2026)

**Tests** (`tests/test_translate.py`, 64 without a download): a fake model for batching, cache, resume after
an interruption, remembered failures, every check, retry and fallback, the aligner with empty translations,
the command line; definition files (errors, fingerprint); prompts, clean-up rules; language codes per family
with a fake tokenizer. One test with a real model (`TRANSLATEALIGN_RUN_MODELS=1`): Opus-MT German-French.

**Opus-MT, end to end** (`Helsinki-NLP/opus-mt-de-fr` and `-fr-de`, RTX 5060 Ti, through the command line
with `tools/model_run.py`; German-French Text+Berg set of Bleualign, scoring as in `docs/RESULTS.md`). Strict
P / R / F1; "source" = the German side translated, "target" = the French side, "both" = intersected:

| set | scorer | source | target | both |
|---|---|---|---|---|
| test document 1 (137 / 155 sentences) | bleu | 0.726 / 0.700 / 0.713 | 0.716 / 0.664 / 0.689 | 0.853 / 0.582 / 0.692 |
| | chrf | 0.759 / 0.773 / 0.766 | 0.784 / 0.791 / 0.787 | 0.827 / 0.736 / 0.779 |
| dev document (468 / 554) | bleu | 0.769 / 0.795 / 0.782 | 0.727 / 0.756 / 0.741 | 0.806 / 0.719 / 0.760 |
| | chrf | 0.762 / 0.808 / 0.785 | 0.783 / 0.835 / 0.808 | 0.813 / 0.798 / 0.805 |
| test set (7 documents, 991 / 1,011) | bleu | 0.855 / 0.837 / 0.846 | 0.850 / 0.825 / 0.837 | 0.924 / 0.775 / 0.843 |
| | chrf | 0.881 / 0.886 / 0.883 | 0.886 / 0.892 / 0.889 | 0.923 / 0.867 / 0.894 |

For comparison, with the translations shipped with Bleualign (RESULTS, command line): test set bleu 0.799 /
0.806 / 0.696, chrf 0.874 / 0.862 / 0.789; dev bleu 0.729, chrf 0.790. Nothing was tuned; these are single
runs on small sets (differences below about 0.005 are noise), the broader evaluation is in docs/RESULTS.md section 3.
Time: 466 German sentences in 12 s, 550 French in 10 s, each including loading the model (not measured
separately; section 6 measures speed and memory). No sentence failed a check, so **the retry path was not
exercised by Opus-MT**; a second run took everything from the cache (0.4 s).

**The chat backend with a real model**: TranslateGemma 4b from the local cache, definition as in section 3, test document 1 in both
directions. First with no stop string and `lines = "keep"`: 5 of 153 French sentences failed the one-line
check twice (all ending in a colon; after blank lines the model repeated its answer or added "Ich kann keine
Informationen über ... finden") and got empty translations; the aligner ran. With `stop = ["\n\n"]` and
`lines = "join"`: no failures, the five answers are the plain translations. Strict F1 on that document: bleu
0.757 (source) / 0.688 (target), chrf 0.745 / 0.776; 136 sentences in 52 s including loading. One observation: the same sentence can get a different answer in a different batch (padding, bfloat16); the cache
keeps the first accepted one.

**Not run in this initial check:** NLLB-200 and MADLAD-400 (only their language handling, with a fake tokenizer; later runs are in section 6);
Opus-MT models with several target languages (`>>fra<<`); `mode = "plain"`, `system`, `response_prefix`,
`template_args` and quantised loading with a real model; any model that needs `trust_remote_code`; CPU and
macOS (`mps`); texts that are not tokenised (the evaluation files are).

## 6. Shipped definitions, speed, memory, licences (14 Nov 2026)

`--model NAME` takes the name of a definition that ships with the package (`src/translatealign/models/NAME.toml`),
a Hugging Face id (seq2seq families `opus-mt`, `nllb`, `madlad`), or `--model-file PATH`. Python:
`load_translator(model="nllb-200")`.

| name | model | licence | languages (as used here) | loaded as |
|---|---|---|---|---|
| `opus-mt` | `Helsinki-NLP/opus-mt-{src}-{trg}` | CC-BY-4.0 | one model per pair; the pair must exist | fp32 |
| `madlad-400` | `google/madlad400-3b-mt` | Apache-2.0 | 400+ languages | bf16 (stored fp32, 15.7 GB download) |
| `nllb-200` | `facebook/nllb-200-distilled-1.3B` | **CC-BY-NC-4.0: non-commercial only** | 200 languages | fp16 |
| `translategemma-4b` | `google/translategemma-4b-it` | Gemma Terms of Use; **gated** (accept the terms on Hugging Face, log in) | 55 languages | bf16 |
| `translategemma-12b` | `google/translategemma-12b-it` | Gemma Terms of Use; gated | 55 languages | 4-bit NF4 (23 GB in bf16 does not fit 16 GB) |
| `qwen-mt-lumen` | `ATH-MaaS/Qwen-MT-Lumen` (14.8 B) | Apache-2.0 | **into** Icelandic (and the other WMT26 targets) from English; not out of Icelandic | 4-bit NF4 (29.5 GB download in bf16) |

Other sizes work with `--model` (`facebook/nllb-200-distilled-600M`, `facebook/nllb-200-3.3B`, MADLAD 7B) or a
copy of a definition file with another `model`. Bigger models were not run.

### Speed and memory

RTX 5060 Ti (16 GB), one process per run, batch sizes as in the definitions, no cache, greedy decoding except
Opus-MT and NLLB (4 beams). Test texts: **de-fr / fr-de** = test document 1 of the Bleualign set (137 German /
155 French sentences, about 114 characters each); **en-is / is-en** = ParIce document `s_1` of the SentAlign 2
evaluation data (162 English / 180 Icelandic sentences, about 25 characters each: very short). "Load" = loading
the model and translating one sentence. GPU memory in GB: after loading / peak during the run. "7,000 sentences"
= minutes for a book, extrapolated linearly from that run (**only the de-fr and fr-de rows are comparable to a
book; the ParIce sentences are 4 to 5 times shorter, so those times are too low for real books**).

| model | text | load (s) | sentences/s | GPU GB | 7,000 sentences (min) | failed checks |
|---|---|---|---|---|---|---|
| opus-mt | de-fr | 3.8 | 50 | 0.3 / 0.7 | 2.3 | 0 |
| opus-mt | fr-de | 4.7 | 70 | 0.3 / 0.7 | 1.7 | 0 |
| opus-mt | en-is | 2.4 | 337 | 0.3 / 0.4 | (0.3) | 0 |
| opus-mt | is-en | 2.3 | 264 | 0.3 / 0.4 | (0.4) | 0 |
| nllb-200 | de-fr | 3.3 | 16 | 2.6 / 4.1 | 7.3 | 0 |
| nllb-200 | en-is | 3.2 | 58 | 2.6 / 3.1 | (2.0) | 0 |
| nllb-200 | is-en | 3.2 | 51 | 2.6 / 3.1 | (2.3) | 0 (1 retried) |
| madlad-400 | de-fr | 5.6 | 15 | 5.5 / 6.5 | 7.7 | 0 |
| madlad-400 | en-is | 5.3 | 55 | 5.5 / 5.7 | (2.1) | 0 |
| madlad-400 | is-en | 5.2 | 69 | 5.5 / 5.7 | (1.7) | 0 |
| translategemma-4b | de-fr | 5.2 | 2.8 | 8.0 / 9.0 | 41 | 0 |
| translategemma-4b | en-is | 5.8 | 7.5 | 8.0 / 8.5 | (16) | 0 |
| translategemma-4b | is-en | 6.1 | 6.5 | 8.0 / 8.5 | (18) | 0 |
| translategemma-12b (4-bit) | de-fr | 13.9 | 0.67 | 7.8 / 9.1 | 174 | 2 (empty) |
| translategemma-12b (4-bit) | en-is | 8.0 | 2.0 | 7.8 / 8.3 | (59) | 0 |
| qwen-mt-lumen (4-bit) | en-is | 9.2 | 3.8 | 9.9 / 10.1 | (31) | 0 |

Times in brackets are from the short ParIce sentences. Load times are with the files in the disk cache (a first
load from a cold disk is slower); the 12b and Lumen figures are for 4-bit loading by bitsandbytes 0.50.2.

Observations:
* Opus-MT is about 3 times faster than NLLB and MADLAD and 18 times faster than TranslateGemma 4b on long
  sentences; the LLM-based models cost 40 minutes (4b) to nearly 3 hours (12b) for one 7,000-sentence book in
  one direction (`--translate both` doubles it). The cache makes a second run free.
* Translation quality is **not** measured here (see docs/RESULTS.md section 3). Two things seen in the outputs: NLLB-200 writes
  Icelandic with the wrong letters in places (`ađ ūađ` for `að það`: ð and þ replaced by đ and ū), which hurts
  n-gram scores against real Icelandic text; and the 12b model, on two German lines of the test document
  (a line of only `__ .` and a table of contents), answered with a request for text or a heading ("Contenu
  :"), so both got an empty translation after the retry.
* Lumen was tried with the prompt of its model card (no guidelines); its Icelandic reads well in the three
  examples looked at, but no more than that was checked.
* 4-bit loading of the two large models worked on this GPU (bitsandbytes 0.50.2, torch 2.15 / CUDA 13.2).

**Not run:** `opus-mt` on any other pair; TranslateGemma 4b and 12b in the directions fr-de (4b, 12b), is-en
(12b); `qwen-mt-lumen` on German-French (it has no such direction) and out of Icelandic (it does not translate
out of Icelandic); NLLB 600M / 3.3B, MADLAD 7B; CPU, macOS; 8-bit loading; a cold first download is not timed.
