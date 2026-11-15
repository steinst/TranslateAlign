# TranslateAlign

**A Bleualign imitation.** Sentence alignment in the style of Bleualign, with modern translation models.

TranslateAlign translates one side of a document pair with a machine translation model, compares the translation
with the sentences of the other side (chrF++ or BLEU), and finds the alignment with the search described in the
Bleualign papers. **This is not the official Bleualign** and contains none of its code: the search is
re-implemented from the publications. The original is at https://github.com/rsennrich/Bleualign.

## Install

Install from GitHub while the first PyPI release is being prepared:

    git clone https://github.com/steinst/TranslateAlign.git
    cd TranslateAlign
    pip install .                            # aligning with a ready-made translation
    pip install ".[models]"                  # also translating with a model (PyTorch, transformers)

Python 3.10 or newer; Linux, Windows and macOS. For development and testing, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Command line

Both texts have one sentence per line. A line `.EOA` in both texts separates documents; the search never
crosses it.

With a translation you already have (`de.translated` = `de` translated into French, line by line):

    translatealign book.de book.fr --translation book.de.translated -o book

With a model that translates here:

    translatealign book.de book.fr --model opus-mt -s de -t fr -o book
    translatealign book.en book.is --model madlad-400 -s en -t is --translate both -o book
    translatealign book.en book.is --model-file my_model.toml -s en -t is -o book

Written: `book.path` (one bead per line, `[0, 1]:[2]` = lines 0 and 1 of the source with line 2 of the target,
counted from 0), `book.src` and `book.trg` (the aligned sentences, line by line). After each run the program
prints a summary and what to cite.

* `--translation-of-target FILE`: a translation of the target text into the language of the source. Each
  translation is one run; with more than one, only the beads found in every run are kept (higher precision,
  lower recall). With a model: `--translate source|target|both`.
* `--scorer chrf` (default: chrF++ in both directions) or `--scorer bleu` (the settings of the papers). chrF++ is
  0.04 to 0.11 strict F1 better on the German-French and English-Icelandic evaluation sets.
* `--model`: a shipped definition (`opus-mt`, `nllb-200`, `madlad-400`, `translategemma-4b`,
  `translategemma-12b`, `qwen-mt-lumen`) or a Hugging Face id of an Opus-MT, NLLB-200 or MADLAD-400 model
  (`"Helsinki-NLP/opus-mt-{src}-{trg}"`). `--model-file`: your own model definition (TOML: prompt, language
  names, generation settings, clean-up rules), described in
  [docs/MODELS.md](https://github.com/steinst/TranslateAlign/blob/main/docs/MODELS.md).
* Translations are cached on disk, so nothing is translated twice and an interrupted run goes on where it
  stopped (`--cache FILE`, `--no-cache`, `--save-translations`). Every output of a model is checked (one line
  per line, not empty, no leaked prompt or commentary, plausible length); a sentence that fails twice gets an
  empty translation and a warning, and the alignment still runs.
* `--unaligned`: also write the sentences that were not aligned. `--gold FILE ...`: score the result against a
  gold alignment. `--bibtex`: print BibTeX for what the run used (alone: `translatealign --bibtex --model
  opus-mt`). `--help` for the rest.

Recommended: `--model opus-mt` where the language pair exists, else `--model madlad-400`; `--translate both` for
higher precision, `source` (default) for higher recall. On the evaluation sets the translation model matters
little ([docs/RESULTS.md](https://github.com/steinst/TranslateAlign/blob/main/docs/RESULTS.md), section 3).
Models are downloaded from Hugging Face and have their own licences (NLLB-200: non-commercial; TranslateGemma:
Gemma terms); a model that needs code from its repository is only loaded with `--trust-remote-code`.

## Python

```python
import translatealign

german = translatealign.read_lines("book.de")
french = translatealign.read_lines("book.fr")

# with a ready-made translation of `german` into French (or several, as a list of lists)
pairs = translatealign.align(german, french, translation=translatealign.read_lines("book.de.translated"))

# or with a model
pairs = translatealign.align(german, french, model="opus-mt", source_language="de", target_language="fr",
                             translate="both")

for p in pairs:
    print(p.src, p.trg, p.kind)          # (0, 1) (2,) anchor      -- indices into the two lists
    print(p.source, "|||", p.target)     # the sentences, joined by a space

print(translatealign.cite(model="opus-mt"))                # what to cite
print(translatealign.cite(model="opus-mt", bibtex=True))   # as BibTeX
```

`align(..., unaligned=True)` also returns the sentences that are in no bead (one side empty, `kind` None);
`scorer="bleu"` uses the settings of the papers. The search, scorers and translators are also usable on their
own: `translatealign.search`, `translatealign.scoring`, `translatealign.translate`.

## How well it works

On the Text+Berg German-French test set, with BLEU and the translation the papers used, it comes within 0.01
strict F1 of the original Bleualign (0.799 against 0.807). With chrF++ and current translation models it reaches
0.87 to 0.90 strict F1 there; SentAlign 2 (LaBSE) is about 0.03 better (0.931) where it covers the language
pair. Details:
[docs/RESULTS.md](https://github.com/steinst/TranslateAlign/blob/main/docs/RESULTS.md).

## What to cite

Please cite the software as:

> Steingrimsson, Steinthor (2026). *TranslateAlign* (version 0.1.2) [Software].
> https://doi.org/10.5281/zenodo.24248223

The program shows this citation and the repository link after each run.
`translatealign --bibtex` prints its BibTeX entry together with the methods used.
Machine-readable metadata is in [CITATION.cff](CITATION.cff).

The program prints what to cite after each run (`--bibtex` for BibTeX): the two papers the search comes from,

* Rico Sennrich and Martin Volk (2010). MT-based Sentence Alignment for OCR-generated Parallel Texts. AMTA 2010.
* Rico Sennrich and Martin Volk (2011). Iterative, MT-based Sentence Alignment of Parallel Texts. NODALIDA 2011.

and Gale and Church (1993) for the length-based step, the similarity score (chrF / chrF++ by Popović, or BLEU
by Papineni et al.) and the translation model you used.

## Licence

Apache License 2.0 (see `LICENSE`). Translation models are downloaded by the user and have their own licences.
