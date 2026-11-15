# Translating with a model

TranslateAlign can use an existing translation file, a dedicated translation
model, or a language model prompted to translate. Each input sentence must produce
one output line in the same order.

## Install and run

From a clone of TranslateAlign:

```sh
pip install ".[models]"
translatealign book.de book.fr --model opus-mt -s de -t fr -o book
translatealign book.en book.is --model madlad-400 -s en -t is --translate both -o book
```

Models are downloaded from Hugging Face on first use. For gated models, accept
the model's terms and authenticate with Hugging Face before running TranslateAlign.
The shipped 12B TranslateGemma and Qwen-MT-Lumen definitions use 4-bit loading;
install `pip install ".[models,quantized]"` and use hardware supported by
bitsandbytes. `--device cpu`, `cuda`, or `mps` overrides automatic device selection;
the model's dtype and quantization must also be supported on that device.

`--translate source` (default) translates the source into the target language.
`target` translates the other side; `both` runs both directions and keeps only
alignments shared by the two runs. Both language directions must be supported by
the selected model. Use `both` for higher precision, or one direction for higher
recall and less translation work. See [RESULTS.md](RESULTS.md) for measured results.

## Choose an approach

| Approach | How it works | How to select it |
|---|---|---|
| Existing translation | Reads translations produced by any external system; no model dependencies needed | `--translation FILE`, optionally `--translation-of-target FILE` |
| Encoder-decoder (`seq2seq`) | Uses a dedicated translation model and its language tokens; no conversational prompt | `--model opus-mt`, `nllb-200`, `madlad-400`, or a supported Hugging Face model ID |
| Decoder (`chat`) | Prompts a language model, using its chat template or a plain-text prompt; output may need cleanup | A shipped chat-model name or `--model-file FILE` |

An existing translation must have one line per input line, including matching
`.EOA` document delimiters. With local models, blank lines and delimiters are
preserved automatically.

### Shipped definitions

The table describes the definitions included with TranslateAlign. Models have
their own licences and language coverage; check the linked model card for the
specific checkpoint you use.

| Name | Checkpoint | Backend and loading | Licence and use |
|---|---|---|---|
| `opus-mt` | [Helsinki-NLP/opus-mt-{src}-{trg}](https://huggingface.co/Helsinki-NLP) | Seq2seq, float32 | CC-BY-4.0 in the shipped definition; a separate checkpoint must exist for each language direction |
| `madlad-400` | [google/madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt) | Seq2seq, bfloat16 | Apache-2.0; multilingual |
| `nllb-200` | [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) | Seq2seq, float16 | CC-BY-NC-4.0; non-commercial |
| `translategemma-4b` | [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) | Chat, bfloat16 | Gemma terms; gated |
| `translategemma-12b` | [google/translategemma-12b-it](https://huggingface.co/google/translategemma-12b-it) | Chat, 4-bit NF4 | Gemma terms; gated |
| `qwen-mt-lumen` | [ATH-MaaS/Qwen-MT-Lumen](https://huggingface.co/ATH-MaaS/Qwen-MT-Lumen) | Chat, 4-bit NF4 | Apache-2.0; supports English into Icelandic, not the reverse |

Opus-MT is a useful starting point when a checkpoint exists for your pair;
MADLAD-400 provides broader language coverage. Larger chat models need more memory
and translation time. Quantization reduces weight memory, but download size and
runtime memory are not the same thing.

### Language codes

Use ISO 639-1 codes such as `de`, `fr`, and `is`, or the model's own language codes.

| Family | Language selection |
|---|---|
| Opus-MT | Substitutes `{src}` and `{trg}` in the model ID; adds a target-language token when the tokenizer requires one |
| NLLB | Maps codes to FLORES-200 codes such as `deu_Latn`; selects the source tokenizer language and target decoder token. Use an explicit code if the script is ambiguous |
| MADLAD | Adds a target-language prefix such as `<2fr>` |
| Chat | Expands language names or codes in the prompt. Use `[languages]` to add or override names |

## Custom model definitions

Copy a [shipped TOML definition](../src/translatealign/models) and pass it with
`--model-file my_model.toml`. `model` is a Hugging Face ID or local directory.
For seq2seq models, `family` is `opus-mt`, `nllb`, or `madlad`; it can be inferred
from a recognizable model ID. Chat models require a prompt. Unknown configuration
keys are rejected.

A seq2seq definition:

```toml
backend = "seq2seq"
model = "facebook/nllb-200-distilled-600M"
family = "nllb"

[generation]
num_beams = 4

[languages]
de = "deu_Latn"
```

A chat definition:

```toml
name = "my-translator"
backend = "chat"
model = "organisation/model-id"

[prompt]
mode = "chat"
template = "Translate the following {source_language} text into {target_language}. Return only the translation.\n{text}"

[generation]
batch_size = 8
max_batch_chars = 8000
max_new_tokens = 256
stop = ["\n\n"]

[retry]
repetition_penalty = 1.2
stop = ["\n"]

[cleanup]
lines = "join"
strip_prefixes = ["Translation:"]

[load]
dtype = "bfloat16"
```

`mode = "chat"` applies the tokenizer's chat template; `mode = "plain"` sends the
formatted prompt directly. Optional fields include `system`, `response_prefix`,
and `template_args` (arguments for the chat template). Available placeholders are
`{text}`, `{source_language}`, `{target_language}`, `{source_code}`, and
`{target_code}`. Language names can be overridden in `[languages]`.

Some models require structured message content instead of a text template.
For example, TranslateGemma uses `[prompt.content]`:

```toml
backend = "chat"
model = "google/translategemma-4b-it"

[prompt]
mode = "chat"

[prompt.content]
type = "text"
source_lang_code = "{source_code}"
target_lang_code = "{target_code}"
text = "{text}"

[generation]
stop = ["\n\n"]

[cleanup]
lines = "join"

[load]
dtype = "bfloat16"
```

### Configuration options

| Table | Options |
|---|---|
| `[generation]` | `batch_size` (default 16), `max_batch_chars` (8000), `max_new_tokens` (256), `new_tokens_per_input_token` (3.0), `max_input_tokens` (0: model limit, seq2seq), and `stop` (list of strings). Other keys, such as `num_beams`, go to the model's `generate()` |
| `[retry]` | Overrides generation settings for the second attempt. For chat models, provide different settings if you want a different retry |
| `[cleanup]` | Chat output: `remove` deletes regex matches; `strip_prefixes` removes leading labels; `lines` is `keep`, `first`, `last`, or `join`; `strip_quotes` removes enclosing quotes if the source has none |
| `[load]` | `dtype`, `device`, `quantization` (`none`, `4bit`, `8bit`), `revision`, `subfolder`, and `model_class` (chat) |
| `[checks]` | Character-length bounds `min_ratio` (1/6), `max_ratio` (6), and `slack` (10); `extra_leak_patterns` adds commentary regexes, while `leak_patterns` replaces the built-in list |

Set `[load] revision` to an immutable Hugging Face commit hash to pin model and
tokenizer weights. A branch or tag can move. The revision is part of the translation
cache key; without a pinned revision, changed remote weights can reuse older cached
translations. For local model directories, use a new cache when replacing weights.

Chat decoding is greedy unless overridden. Stop strings cut the answer before
cleanup runs, so avoid stopping at the first newline if the model produces
multiline reasoning that cleanup must remove. Keep `lines = "keep"` when unexpected
extra lines should fail validation rather than be discarded or joined.

A model requiring custom repository code must declare `trust_remote_code = true`
in its definition and be run with `--trust-remote-code`.

## Caching, warnings, and memory

Set `--cache FILE`, use `$TRANSLATEALIGN_CACHE`, or accept the default
`~/.cache/translatealign/translations.sqlite`. `--no-cache` disables caching;
`--save-translations` writes the translated texts alongside the alignment.
Completed batches are saved so an interrupted run can resume.

Outputs are checked for line count, empty translations, leaked prompts, commentary,
and implausible length. A failed sentence is retried once, then receives an empty
translation and a warning if it still fails. Failures are remembered;
`--retry-failed` retries them. Cached successes are rechecked under the current
rules, and changing the rules also allows cached failures to be retried.

If generation reaches its token limit, TranslateAlign warns that the translation
may be incomplete and does not cache it. Raise `max_new_tokens` and, if needed,
`new_tokens_per_input_token`. Seq2seq input can also be truncated at the tokenizer's
input limit; this does not produce the same warning. Validation cannot guarantee
accuracy or completeness, and built-in commentary checks mainly cover English.

If a model runs out of memory, reduce `batch_size` and `max_batch_chars` in a copy
of its definition, select a smaller model, or use supported quantization. When
switching device, adjust `[load] dtype` and quantization as needed.

## Python

```python
from translatealign.translate import load_translator, TranslationCache

translator = load_translator(model="opus-mt")  # or model_file="my_model.toml"
with TranslationCache() as cache:
    french = translator.translate(["Guten Morgen."], "de", "fr", cache=cache)

print(french)            # a list of translations
print(french.problems)   # sentences that failed validation twice
```
