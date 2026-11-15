# TranslateAlign – results

## 1. Reproduction of the Bleualign table (14 Nov 2026)

Data: the German-French Text+Berg evaluation set shipped with Bleualign (test set `eval1989`: 7 documents,
991 / 1,011 sentences, 858 gold beads with sentences on both sides; development document `eval1957`: 468 / 554
sentences, 381 gold beads) and the ready-made translations shipped with it. Everything below is made by

    TRANSLATEALIGN_BLEUALIGN_REPO=/path/to/Bleualign python tools/reproduce.py        (8 seconds)

which calls the command line (`translatealign eval1989.de eval1989.fr --translation eval1989.europarlfull.fr
-o OUT`, with `--translation-of-target eval1989.europarlfull.de` for the intersected run), reads `OUT.path`
and scores it with `translatealign.evaluate`.

**Scoring** (`src/translatealign/evaluate.py`, the convention of SPEC section 6): only beads with sentences
on both sides count; strict = identical to a gold bead; lax = overlapping a gold bead on both sides (precision
over the tested beads, recall over the gold beads). Applied to the original program's own output it gives
exactly the numbers that program prints, in all five runs (first table; a test checks this). The strict numbers
are also those of the SentAlign / Vecalign script (`SentAlign2/evaluation/evaluate.py`, checked on two runs:
0.816 / 0.782 / 0.799 and 0.879 / 0.869 / 0.874). **Its lax recall is defined differently and comes out
lower** (0.812 instead of 0.935 on the first run; lax F1 0.889 instead of 0.958): lax numbers of the two
conventions must not be compared. That script was not copied; the comparison in section 3 uses one convention for both
aligners.

### The original Bleualign (its output, scored here; equal to its own report and to [SV10] table 4)

| run | beads | strict P / R / F1 | lax P / R / F1 |
|---|---|---|---|
| test, `europarlfull` | 813 | 0.829 / 0.786 / 0.807 | 0.978 / 0.921 / 0.948 |
| test, `google` | 813 | 0.831 / 0.788 / 0.809 | 0.978 / 0.923 / 0.950 |
| test, `europarllight` | 723 | 0.722 / 0.608 / 0.660 | 0.906 / 0.769 / 0.832 |
| test, both directions intersected (`europarlfull`) | 644 | 0.916 / 0.688 / 0.786 | 0.986 / 0.734 / 0.842 |
| dev, `europarlfull` | 374 | 0.749 / 0.735 / 0.742 | 0.987 / 0.929 / 0.957 |

### TranslateAlign, `--scorer bleu` (command line; standard Gale-Church from the `galechurch` package)

| run | beads | strict P / R / F1 | lax P / R / F1 | beads identical to the original's |
|---|---|---|---|---|
| test, `europarlfull` | 822 | 0.816 / 0.782 / 0.799 | 0.983 / 0.935 / 0.958 | 770 / 813 |
| test, `google` | 821 | 0.825 / 0.789 / 0.806 | 0.984 / 0.932 / 0.958 | 761 / 813 |
| test, `europarllight` | 745 | 0.749 / 0.650 / 0.696 | 0.921 / 0.794 / 0.853 | 651 / 723 |
| test, both directions intersected (`europarlfull`) | 632 | 0.921 / 0.678 / 0.781 | 0.991 / 0.725 / 0.837 | 604 / 644 |
| dev, `europarlfull` | 376 | 0.734 / 0.724 / 0.729 | 0.989 / 0.927 / 0.957 | 351 / 374 |

### TranslateAlign, BLEU, with the Gale-Church of the original for the gaps (library, recorded output; not the command line)

This separates the search from the length aligner (SPEC 7.6: the original's Gale-Church is not the standard
algorithm, and TranslateAlign does not imitate it).

| run | beads | strict P / R / F1 | lax P / R / F1 | beads identical to the original's |
|---|---|---|---|---|
| test, `europarlfull` | 815 | 0.823 / 0.782 / 0.802 | 0.977 / 0.920 / 0.947 | 804 / 813 |
| test, `google` | 815 | 0.829 / 0.788 / 0.808 | 0.977 / 0.920 / 0.947 | 795 / 813 |
| test, `europarllight` | 731 | 0.740 / 0.631 / 0.681 | 0.914 / 0.774 / 0.838 | 700 / 723 |
| test, both directions intersected (`europarlfull`) | 641 | 0.917 / 0.685 / 0.785 | 0.986 / 0.731 / 0.839 | 637 / 644 |
| dev, `europarlfull` | 376 | 0.739 / 0.730 / 0.734 | 0.987 / 0.927 / 0.956 | 361 / 374 |

### Reading of the BLEU tables
* **Identical beads: as in SPEC section 7.9** (770, 761, 651, 351, 604 with the standard Gale-Church; 804, 795, 700, 361,
  637 with the original's), now reached end to end through the command line.
* **Target "the same scores to two decimals": met in part.** Strict F1 differs from the original by -0.008,
  -0.003, +0.036, -0.005 and -0.013 (command line), and by -0.005, -0.001, +0.021, -0.001 and -0.008 with the
  original's Gale-Church. Rounded to two decimals, strict F1 is the same in one run of five on the command
  line (`google`, 0.81) and in two with the original's Gale-Church (`google`; intersected, 0.79). Lax F1 is
  within 0.010 except `europarllight` (+0.021).
* What the differences are: with good translations the 1 to 3 % of beads that the search does not reproduce
  (SPEC 7.9, unexplained) cost up to 0.005 strict F1 on the test set (the size of the noise on this set) and
  0.008 on the development document; the standard Gale-Church costs another 0.002 to 0.005 strict and gains
  about 0.01 lax. With the weak translation (`europarllight`) TranslateAlign is **better** than the original
  (0.696 against 0.660), both parts contributing: more sentences reach the length aligner and the standard
  algorithm aligns them better.
* Not tested: other language pairs; texts that are not tokenised (the evaluation files are).

## 2. chrF++ on the same files 

`--scorer chrf` is chrF++ (character n-grams to 6, word n-grams to 2, beta = 2, lowercased) computed **in both
directions and combined by the harmonic mean**, as the papers do for BLEU. The search and its options are
unchanged (those found for BLEU during reference reproduction).

How that was chosen: plain chrF++ (the translation as hypothesis, one direction) was run first, on all files.
Its strict scores were poor, and the reason is in the measure: with beta = 2 it rewards recall, so a joined
sequence nearly always scores higher, and the extra condition "more matches" is always true for character
n-grams. The alternatives were then compared **on the development document only** (one direction or both;
beta 2 or 1; with and without word n-grams; the match condition on the target side): every symmetric variant
gives the same alignment there (strict F1 0.790, against 0.632 for the plain one and 0.729 for BLEU), so the
one closest to the papers was taken. Nothing was chosen on the test set; the test numbers of the plain variant
are shown because they were seen before the choice.

| run | beads | strict P / R / F1 | lax P / R / F1 | BLEU strict F1 (above) |
|---|---|---|---|---|
| test, `europarlfull` | 849 | 0.879 / 0.869 / **0.874** | 0.994 / 0.973 / 0.984 | 0.799 |
| test, `google` | 859 | 0.861 / 0.862 / **0.862** | 0.988 / 0.976 / 0.982 | 0.806 |
| test, `europarllight` | 798 | 0.818 / 0.761 / **0.789** | 0.965 / 0.893 / 0.927 | 0.696 |
| test, both directions intersected (`europarlfull`) | 767 | 0.922 / 0.824 / **0.870** | 0.997 / 0.883 / 0.937 | 0.781 |
| dev, `europarlfull` | 399 | 0.772 / 0.808 / **0.790** | 0.997 / 0.992 / 0.995 | 0.729 |

Plain chrF++, one direction (library only, not offered on the command line), strict P / R / F1: 0.781 / 0.752 /
0.766, 0.820 / 0.814 / 0.817, 0.673 / 0.584 / 0.625, 0.927 / 0.573 / 0.708 (intersected), dev 0.631 / 0.633 /
0.632.

* chrF++ in both directions is 0.06 to 0.09 strict F1 above BLEU on every test run, far outside the noise, and
  loses much less recall when two directions are intersected.
* For comparison (section 3 compares modern translation models): SentAlign 2 with LaBSE has strict
  F1 0.931 on this test set.
* Established in section 3.1: on the development document the search options make no difference for chrF++,
  and chrF++ is now the default scorer of the command line.

## 3. Evaluation with translation models, against SentAlign 2 (15 Nov 2026)

**Question:** how good is TranslateAlign with present-day translation models, per model and scorer, and how does it
compare with SentAlign 2 (LaBSE) and with the original Bleualign?

**Data.** German-French: the Bleualign / Text+Berg set, *test* = `eval1989` (7 documents, 858 gold beads with
sentences on both sides) and *dev* = `eval1957` (1 document, 381). English-Icelandic: ParIce, the 10 documents of
the SentAlign 2 evaluation data (515 gold beads). **Settings were chosen on the German-French dev document only**;
there is no English-Icelandic dev set, so English-Icelandic is a held-out pair. Scoring: `translatealign.evaluate`
for both aligners (SentAlign 2's stored evaluation output is re-scored here and gives its published
0.931 / 0.951 exactly). Strict F1 is from counts summed over the documents; intervals: 95 % percentile bootstrap
over documents (10,000 resamples). With 7 and 10 documents the intervals are wide.

**Translation:** each side of each set translated once per model (shipped definitions in docs/MODELS.md, greedy except
Opus-MT and NLLB with 4 beams), then aligned with each scorer: translation of the source side, of the target side,
and the two runs intersected (`--translate source | target | both`). `qwen-mt-lumen` only translates English into
Icelandic, so it has the source-side run on ParIce only.

    python tools/translate_sets.py MODEL                         # translations to $TRANSLATEALIGN_EVAL7
    python tools/evaluate_all.py --sets de-fr-dev de-fr-test en-is --configs bleu chrf --out all.json
    python tools/sentalign_baseline.py sentalign.json --root /path/to/stored-output             # SentAlign 2's stored output, same scoring
    python tools/report.py all.json sentalign.json                # the tables below, with intervals
    python tools/analyse.py                                       # 3.3 to 3.5

(The texts, translations and gold files stay outside the repository: GPL-2.0 / CC BY 4.0.)

### 3.1 Settings, chosen on the dev document

* **Scorer: chrF++ in both directions** (as in section 2). With every one of the five models it beats BLEU on dev
  (by 0.003 to 0.11 strict F1 for the same run, about 0.05 on average) and, as it later turned out, on both test
  sets by 0.04 to 0.11 for every run.
* **Search options: unchanged.** With chrF++ on dev (Opus-MT, NLLB, MADLAD, all three runs) none of beta = 1,
  the match condition on the target side or dropped on the source side, `max_sentences = 4`, k = 2 or 5, or the
  one-to-one heuristic switched off moved the mean strict F1 of a run by more than 0.004 (k = 2 on the target
  side; +0.002 to -0.001 elsewhere), and none helped all three runs. Two choices are clearly worse and confirm the
  existing ones: `max_items = 3` (-0.06 to -0.07) and `max_items = 1` (-0.17).
* **Run: both directions intersected** was the best of the three on dev for four of five models (mean 0.806,
  against 0.792 one side, 0.797 the other); it was not used to choose anything on the test sets.

### 3.2 Main table (strict, chrF++)

| model | de-fr test: source / target / **both** | both: P / R | interval (both) | en-is: source / target / **both** | both: P / R | interval (both) |
|---|---|---|---|---|---|---|
| `opus-mt` | 0.883 / 0.889 / **0.894** | 0.923 / 0.867 | 0.853 - 0.924 | 0.902 / 0.910 / **0.912** | 0.944 / 0.882 | 0.850 - 0.946 |
| `nllb-200` | 0.871 / 0.886 / **0.886** | 0.921 / 0.854 | 0.846 - 0.916 | 0.900 / 0.900 / **0.901** | 0.943 / 0.862 | 0.852 - 0.928 |
| `madlad-400` | 0.884 / 0.894 / **0.901** | 0.933 / 0.871 | 0.874 - 0.924 | 0.902 / 0.921 / **0.915** | 0.948 / 0.883 | 0.866 - 0.944 |
| `translategemma-4b` | 0.854 / 0.879 / **0.872** | 0.929 / 0.822 | 0.830 - 0.906 | 0.875 / 0.889 / **0.883** | 0.945 / 0.829 | 0.848 - 0.910 |
| `translategemma-12b` (4-bit) | 0.879 / 0.889 / **0.891** | 0.939 / 0.847 | 0.855 - 0.919 | 0.902 / 0.902 / **0.901** | 0.956 / 0.852 | 0.859 - 0.930 |
| `qwen-mt-lumen` (4-bit) | - | | | 0.905 / - / - | (source) 0.902 / 0.909 | 0.869 - 0.930 |
| **SentAlign 2, LaBSE** | **0.931** | 0.934 / 0.928 | 0.908 - 0.953 | **0.951** | 0.948 / 0.953 | 0.914 - 0.972 |
| original Bleualign (`europarlfull`, section 1) | 0.807 | 0.829 / 0.786 | | not run | | |
| TranslateAlign, best BLEU run of any model | 0.847 | | | 0.879 | | |

Full tables (BLEU, lax scores, beads, every run, the paired difference to SentAlign 2 with its interval): `python
tools/report.py`. Lax F1 of the chrF++ runs: 0.98 to 0.99 for one side, 0.93 to 0.96 intersected; SentAlign 2
0.989 / 0.986.

**Reading.**
* **TranslateAlign stays below SentAlign 2, on both pairs, with every model.** The best runs are 0.030 below it
  on both sets (MADLAD: intersected on de-fr, target side on en-is); the paired bootstrap interval of the
  difference excludes zero for every model and every run on both sets (best runs: -0.040 to -0.024 on de-fr,
  -0.066 to -0.002 on en-is).
* **Against the original Bleualign it is a large step:** 0.894 to 0.901 against 0.807 on the same test set
  (+0.09), of which chrF++ gives about 0.05 and the better translation the rest (Opus-MT with BLEU: 0.846).
* **The translation model matters little.** On de-fr test the five models span 0.872 to 0.901, on en-is 0.883 to
  0.915. Paired bootstrap (chrF++, both): Opus-MT cannot be told apart from MADLAD or TranslateGemma 12b on either
  set; MADLAD is better than NLLB on both sets (+0.015 / +0.014) and than 12b on en-is (+0.013); TranslateGemma 4b
  is the worst on both (significantly below MADLAD, 12b and, on de-fr, Opus-MT and NLLB). The 23-GB 12b model is
  no better than the 300-MB Opus-MT.
* **Intersecting buys precision with recall.** The intersected runs reach SentAlign 2's precision (0.92 to 0.96
  against 0.93 / 0.95; TranslateGemma 12b has 0.956 on en-is) at a recall 0.06 to 0.12 lower.

### 3.3 Where the difference to SentAlign 2 is: beads that are not one-to-one

Recall per shape of the gold bead (strict; numbers found / gold):

| | de-fr 1-1 (678) | de-fr 2-1 (82) | de-fr 1-2 (63) | de-fr larger (35) | en-is 1-1 (443) | en-is 2-1 / 1-2 (61) | en-is larger (11) |
|---|---|---|---|---|---|---|---|
| SentAlign 2 | 0.978 | 0.829 | 0.825 | 0.371 | 0.975 | 0.820 | 0.818 |
| Opus-MT, source | 0.973 | 0.683 | 0.651 | 0.086 | 0.973 | 0.574 | 0.273 |
| Opus-MT, both | 0.969 | 0.598 | 0.587 | 0.029 | 0.962 | 0.426 | 0.182 |
| MADLAD, source | 0.979 | 0.610 | 0.683 | 0.086 | 0.957 | 0.689 | 0.182 |
| MADLAD, both | 0.975 | 0.549 | 0.619 | 0.057 | 0.950 | 0.541 | 0.091 |

* **On one-to-one beads TranslateAlign is as good as SentAlign 2** (recall 0.95 to 0.98 against 0.975 to 0.978).
* **The gap is the one-to-many beads.** Of the gold beads SentAlign 2 finds and Opus-MT (both) does not, 83 %
  (de-fr, 48 of 58) and 77 % (en-is, 34 of 44) are not one-to-one (the other way round there are only 6 and 7
  such beads). In two thirds of the 58 / 44 (39 / 29) a
  sentence of the bead is left unaligned altogether: one run joined it, the other did not, and the intersection
  drops it; one direction alone finds more of them but pays with wrong 1-1 beads (1-1 precision 0.90 to 0.91 one
  side, 0.94 to 0.96 intersected, SentAlign 2 0.94 / 0.95).
* Per search stage (Opus-MT, both, de-fr / en-is): first-pass anchors are right 94 % / 95 %, the gap heuristics
  86 % / 95 %, Gale-Church 67 % / 85 % (12 and 26 beads only).
* None of the search options of 3.1 moves this; it is in the design of the search (one-to-one anchors from the first
  pass, joining only by local heuristics in the gaps), not in the translation. A better translation model will
  not close it (3.4). Closing it needs a search that weighs 1-2 / 2-1 against 1-1 everywhere (as SentAlign's
  does): the "plug-in" item of the plan, or a change of the search that would make it no longer Bleualign's.

### 3.4 Translation quality and alignment quality

Corpus chrF++ of each translation against the target sentences of the gold one-to-one beads (a rough measure of
the translation, not of a reference translation) and strict F1 of the run with that translation (chrF++):

| model | de-fr test: chrF++ of de→fr / fr→de | F1 source / target | en-is: chrF++ of en→is / is→en | F1 source / target |
|---|---|---|---|---|
| `opus-mt` | 43.3 / 40.3 | 0.883 / 0.889 | 40.2 / 45.2 | 0.902 / 0.910 |
| `nllb-200` | 44.7 / 40.4 | 0.871 / 0.886 | 41.2 / 44.6 | 0.900 / 0.900 |
| `madlad-400` | 44.4 / 42.9 | 0.884 / 0.894 | 42.6 / 48.3 | 0.902 / 0.921 |
| `translategemma-4b` | 41.4 / 38.3 | 0.854 / 0.879 | 32.1 / 42.4 | 0.875 / 0.889 |
| `translategemma-12b` | 44.4 / 42.0 | 0.879 / 0.889 | 36.6 / 45.2 | 0.902 / 0.902 |
| `qwen-mt-lumen` | - | - | 44.7 / - | 0.905 / - |

The model with clearly weaker translations (TranslateGemma 4b, above all into Icelandic: 32.1) also aligns worst;
among the others, 2 to 4 chrF++ points of translation quality make no difference to the alignment that this set
can measure. NLLB's Icelandic with đ / ū for ð / þ (docs/MODELS.md section 6) costs it nothing visible here (41.2 into Icelandic,
F1 0.900). Lumen, the best translation into Icelandic (44.7), gives the best source-side F1 on en-is (0.905), within
the noise of the others.

### 3.5 Failed translations, and a fault in the checks found and fixed

* The first runs showed that the **initial commentary check rejected correct translations**: `^(sure|of course|
  okay|ok)... (here|I ...)` and `I cannot / am unable to ... (help|provide)` match ordinary sentences such as
  "Of course I am ready to keep my promise." (from Icelandic "Auðvitað ...") or "I am unable to provide you with
  what you are looking for." The rule "only if the source does not match too" cannot work across languages. 15 of
  the 16 commentary rejections in this evaluation were such sentences (they got an empty translation). The two patterns
  now ask for words about the translation itself ("Of course, here is the translation:", "Okay, let me
  translate", "I cannot provide a translation"); tests for both sides (`tests/test_translate.py`). The affected
  sentences were translated again (`translate_sets.py --retry-failed`) and all tables above are after the fix; it
  changed en-is F1 by at most +0.004 and nothing on de-fr.
* What is left empty after the fix is right to be: TranslateGemma 12b answers OCR debris lines (`!`, `.....`,
  `__ .`, `JfV`) of the German-French set with "Veuillez fournir le texte allemand à traduire" (26 lines over all
  files; the length check catches it) and a table-of-contents line with "Contenu :", NLLB turns `!` into "- Ich weiß nicht.", and one real chat preamble
  ("Certainly. Here's the English translation ..."). Filling these lines from Opus-MT changes F1 by -0.002 to
  +0.001: the aligner copes with a few empty translations as intended.

### 3.6 Time and memory: two books

English-Icelandic books of the SentAlign 2 regression set, each aligned **as one document** (no chapter
delimiters, the largest problem for the search), Opus-MT, chrF++, through the command line, RTX 5060 Ti, 20 cores:

| book | sentences en / is | `--translate source`: wall time | `both`: wall time (source side from the cache) | search only, both directions (all cached): chrF++ / BLEU | peak memory: chrF++ / BLEU |
|---|---|---|---|---|---|
| Pride and Prejudice | 6,094 / 6,716 | 95 s | 120 s | 50 s / 13 s | 2.4 GB / 0.6 GB |
| Harry Potter and the Chamber of Secrets | 6,914 / 6,856 | 82 s | 113 s | 47 s / 14 s | 2.2 GB / 0.6 GB |

So a 7,000-sentence book takes about 1.5 minutes in one direction and 3.3 to 3.6 minutes from scratch with
`--translate both` with Opus-MT, about a quarter of it in the chrF++ search. The search is quadratic in the length
of a document: a book three times as long (the Forsyte Saga, 22,000 sentences) in one piece would need, by
extrapolation (not run), some 20 to 25 GB and 8 minutes for the search: split long books at chapters with `.EOA`
lines. The book runs were made before the fix of 3.5 (5 and 6 sentences of the Icelandic side were left empty);
the intersected runs after the fix change the numbers below by at most 0.001. Against SentAlign 2's own output for these two books: 97 % of
TranslateAlign's intersected beads are also SentAlign 2's, and they cover 88 % / 91 % of SentAlign 2's beads
(source side only: 93 % / 94 % both ways), the same pattern as on the gold sets.

After performance improvements that preserved the output the search of Pride and Prejudice takes about 35 s with chrF++ and 9 s with BLEU for both directions (peak memory 1.0 GB / 0.3 GB); the figures above were measured before.

Translation time for a 7,000-sentence book, one direction, from the German-French test set (sentences of about
115 characters, book-like; model loading not included):

| model | sentences / s | minutes per 7,000 sentences |
|---|---|---|
| `opus-mt` | 66 - 75 | 1.6 - 1.8 |
| `nllb-200` | 17 - 18 | 6.4 - 6.9 |
| `madlad-400` | 17 - 18 | 6.6 - 7.0 |
| `translategemma-4b` | 3.4 - 3.5 | 33 - 34 |
| `translategemma-12b` (4-bit) | 0.8 | 146 - 148 |
| `qwen-mt-lumen` (4-bit, en→is) | 2.1 (on ParIce's short sentences; slower on books) | more than 56 |

Not measured: SentAlign 2 on the same books here (its own report: LaBSE encoding is 94 to 99 % of its time); a
book with chapter delimiters; other models on books.

### 3.7 Recommendation

* **Default scorer: chrF++ (`--scorer chrf`), now the default of the command line and of `cli.align_files`**
  (chosen on dev, confirmed on both test sets for every model). `--scorer bleu` stays for the papers' settings.
* **Default translation model: Opus-MT** where the language pair exists. It is as good as any model tried
  (statistically indistinguishable from the best on both sets), 4 times faster than NLLB / MADLAD and 20 to 90
  times faster than the LLM-based models, 300 MB, CC-BY-4.0. **MADLAD-400** (Apache-2.0) for pairs that Opus-MT does
  not have: numerically the best on both sets. Not recommended as a default: NLLB-200 (non-commercial licence, not
  better), TranslateGemma (gated, 20 to 90 times slower, 4b clearly worse, 12b not better), Lumen (one direction
  only; fine for English into Icelandic, but not better than Opus-MT for alignment).
* **Run: `--translate both`** when precision matters (0.92 to 0.96 precision; best or within 0.007 of the best on
  dev and both test sets) and the model translates both ways; `--translate source` (the default, which every model
  can do) for higher recall at lower precision. The default is left as it is because `both` doubles the
  translation time and needs a model for each direction.
* **Against SentAlign 2:** where a good sentence encoder (LaBSE) covers the language pair, SentAlign 2 is the
  better aligner (+0.03 to +0.04 strict F1, outside the noise), because it handles one-to-many beads (3.3).
  TranslateAlign's place is where a translation system exists but no good encoder does, or where a precise,
  inspectable set of mostly one-to-one beads is wanted (precision equal to SentAlign 2's with `both`).

**Not tested:** other language pairs; a dev set for English-Icelandic (none exists); untokenised German-French
text (the set is tokenised, ParIce is not); NLLB 3.3B, MADLAD 7B; 8-bit loading; COMET-Kiwi as a scorer;
combining the translations of two models.
