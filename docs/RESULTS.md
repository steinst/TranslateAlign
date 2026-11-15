# TranslateAlign – results

Alignment results for version 0.1.2, evaluated on 15 November 2026. Scores use the
current alignment code and stored translations.

**Strict precision, recall and F1** require an exact match to a gold alignment bead
(a pair of sentence groups). Beads with an empty side are excluded. All aligners
are scored with `translatealign.evaluate`. Its lax scores use the Bleualign
convention and should not be compared with the differently defined lax scores of
the SentAlign / Vecalign evaluation script.

## 1. Comparison with the original Bleualign

German-French Text+Berg: the `eval1989` test set has 7 documents, 991 German and
1,011 French sentences, and 858 non-empty gold beads. The `eval1957` development
document has 468 German and 554 French sentences, and 381 non-empty gold beads.
Both programs use the same translations supplied with Bleualign. TranslateAlign
uses `--scorer bleu` and the standard Gale-Church length aligner.

| Set and translation | Original Bleualign: strict F1 | TranslateAlign: strict P / R / F1 |
|---|---|---|
| Test, `europarlfull` | 0.807 | 0.816 / 0.782 / **0.799** |
| Test, `google` | 0.809 | 0.825 / 0.789 / **0.806** |
| Test, `europarllight` | 0.660 | 0.749 / 0.650 / **0.696** |
| Test, `europarlfull`, both directions intersected | 0.786 | 0.921 / 0.678 / **0.781** |
| Development, `europarlfull` | 0.742 | 0.734 / 0.724 / **0.729** |

TranslateAlign is an independent implementation of the published method, not an
exact reproduction of the original program's output.

## 2. Default scorer: chrF++

`--scorer chrf` uses character n-grams up to 6 and word n-grams up to 2, beta = 2,
and lowercased text. Each sentence-pair score is computed in both directions and
combined by the harmonic mean. This is separate from `--translate both`, which
intersects alignments made with translations in the two language directions.

With the same supplied translations as in section 1:

| Set and translation | chrF++: strict P / R / F1 | BLEU: strict F1 |
|---|---|---|
| Test, `europarlfull` | 0.879 / 0.869 / **0.874** | 0.799 |
| Test, `google` | 0.861 / 0.862 / **0.862** | 0.806 |
| Test, `europarllight` | 0.818 / 0.761 / **0.789** | 0.696 |
| Test, `europarlfull`, both directions intersected | 0.922 / 0.824 / **0.870** | 0.781 |
| Development, `europarlfull` | 0.772 / 0.808 / **0.790** | 0.729 |

## 3. Results with translation models

German-French uses the Text+Berg test set above. English-Icelandic uses 10 ParIce
documents from the SentAlign 2 evaluation data, with 515 non-empty gold beads.
Settings were selected on the German-French development document; there is no
English-Icelandic development set in this evaluation.

Each side was translated using the [shipped model definitions](MODELS.md), with
greedy decoding except for Opus-MT and NLLB (4 beams). All TranslateAlign rows
below use chrF++. **Source** translates the source text, **target** translates the
target text, and **both** keeps only beads found in both alignment runs.

F1 is calculated from counts summed over documents. The 95% intervals use 10,000
bootstrap resamples of whole documents. With only 7 and 10 documents, the
intervals are wide and small differences between models should not be overinterpreted.

### German-French

| Model | Source F1 | Target F1 | Both F1 | Both: P / R | 95% interval, both |
|---|---|---|---|---|---|
| `opus-mt` | 0.883 | 0.889 | **0.894** | 0.923 / 0.867 | 0.853–0.924 |
| `nllb-200` | 0.871 | 0.886 | **0.886** | 0.921 / 0.854 | 0.846–0.916 |
| `madlad-400` | 0.884 | 0.894 | **0.901** | 0.933 / 0.871 | 0.874–0.924 |
| `translategemma-4b` | 0.854 | 0.879 | **0.872** | 0.929 / 0.822 | 0.830–0.906 |
| `translategemma-12b` (4-bit) | 0.879 | 0.889 | **0.891** | 0.939 / 0.847 | 0.855–0.919 |

SentAlign 2 (LaBSE): **F1 0.931**, precision 0.934, recall 0.928; 95% interval
0.908–0.953. Original Bleualign with `europarlfull`: F1 0.807.

### English-Icelandic

| Model | Source F1 | Target F1 | Both F1 | Both: P / R | 95% interval, both |
|---|---|---|---|---|---|
| `opus-mt` | 0.902 | 0.910 | **0.912** | 0.944 / 0.882 | 0.850–0.946 |
| `nllb-200` | 0.900 | 0.900 | **0.901** | 0.943 / 0.862 | 0.852–0.928 |
| `madlad-400` | 0.902 | 0.921 | **0.915** | 0.948 / 0.883 | 0.866–0.944 |
| `translategemma-4b` | 0.875 | 0.889 | **0.883** | 0.945 / 0.829 | 0.848–0.910 |
| `translategemma-12b` (4-bit) | 0.902 | 0.902 | **0.901** | 0.956 / 0.852 | 0.859–0.930 |

`qwen-mt-lumen` (4-bit), English into Icelandic only: **source F1 0.905**,
precision 0.902, recall 0.909; 95% interval 0.869–0.930.

SentAlign 2 (LaBSE): **F1 0.951**, precision 0.948, recall 0.953; 95% interval
0.914–0.972. Its stored outputs were re-scored with the same evaluation function.

On these two language pairs, the best TranslateAlign runs are about 0.03 F1 below
SentAlign 2. Intersecting the two directions increases precision at the cost of
recall, particularly for beads joining multiple sentences. Opus-MT and MADLAD-400
have similar results within the uncertainty of these small evaluation sets.
These results do not establish performance on other language pairs or domains.

## 4. Search time and memory

*Pride and Prejudice*, English-Icelandic, 6,094 / 6,716 sentences aligned as one
document on a 20-core machine. Both language directions are aligned using stored
translations; translation generation is excluded. Times are from two runs and
vary with machine load.

| Scorer | Search time, both directions | Peak memory |
|---|---|---|
| chrF++ | 35–38 s | 0.99 GB |
| BLEU | 8.6–8.7 s | 0.32 GB |

Search work grows quadratically with document length. For long books, use matching
`.EOA` delimiters at corresponding chapter boundaries. End-to-end runtime also
depends on the translation model, hardware, sentence lengths, and cache reuse.

## 5. Reproducing the results

Set the data paths described in [CONTRIBUTING.md](../CONTRIBUTING.md). Evaluation
texts, gold alignments, and generated translations are not included in this
repository. The reference fixtures contain sentence numbers only.

```sh
# Supplied Bleualign translations: sections 1 and 2.
python tools/reproduce.py

# Repeat for each model, then score all saved translations.
python tools/translate_sets.py MODEL
python tools/evaluate_all.py --sets de-fr-dev de-fr-test en-is --configs bleu chrf --out all.json
python tools/sentalign_baseline.py sentalign.json --root /path/to/stored-output
python tools/report.py all.json sentalign.json
```

The report includes precision, recall, strict and lax F1, and bootstrap intervals
for each run. Models and evaluation datasets have their own licences.
