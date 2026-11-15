# Reference output of the original Bleualign

These files are what the original Bleualign program returns on its own evaluation data. TranslateAlign's search is
tested against them, stage by stage. They contain **sentence numbers only, no text**.

## Where they are from
* Program: Bleualign, https://github.com/rsennrich/Bleualign, commit `5a11b7f` (18 March 2021), by Rico Sennrich;
  run unchanged, as a black box (its source code was not read for TranslateAlign).
* Data: the evaluation files shipped in that repository (`eval/`): the German-French Text+Berg test set
  `eval1989` (7 documents, 991 / 1,011 sentences), the development document `eval1957` (468 / 554 sentences), and
  the machine translations provided with them.
* Command, in a clone of that repository (`-d` instead of `-e` for the development document):

      python3 bleualign.py -s eval/eval1989.de -t eval/eval1989.fr --srctotarget eval/eval1989.europarlfull.fr \
          -o OUT --printempty -e -v 2 > OUT.log

  and `tools/extract_reference.py OUT` turns the log into the JSON file.

| file | data | translation used |
|---|---|---|
| `test_europarlfull.json` | test set | `eval1989.europarlfull.fr` (German translated to French) |
| `test_google.json` | test set | `eval1989.google.fr` |
| `test_europarllight.json` | test set | `eval1989.europarllight.fr` |
| `test_europarlfull_both_directions.json` | test set | `europarlfull` in both directions, intersected (anchors and beads of the combined run; no counts per type) |
| `dev_europarlfull.json` | development document | `eval1957.europarlfull.fr` |

Format: a list with one entry per document, in file order: `anchors` (the 1-to-1 pairs `[source, target]` after
the first pass), `beads` (the final alignment, `[[source numbers], [target numbers]]`), `types` (how many beads
the program attributes to its first pass `BLEU`, its gap heuristics `GAPFILLER` and `GALECHURCH`), `summary`
(the program's own lines on how many sentences were aligned). Sentences are numbered from 0 within a document.

## `galechurch_gaps.json`
What the Gale-Church mode of the same program (`--galechurch --srctotarget -`, run as a black box on the text of
one gap at a time) returns for every gap that TranslateAlign's second pass hands to its length aligner on the runs
above and on the test set in the other direction (French translated to German). Keys: SHA-1 of the gap's text;
values: beads as `[[source numbers], [target numbers]]`, counted from 0 within the gap. No text. Made by
`tools/record_gc_oracle.py`; re-run it if the search changes which gaps reach the length aligner. It lets the
tests compare the heuristics with the original apart from the length aligner, whose behaviour in the original
is not the standard algorithm (docs/SPEC.md 7.6).

## What to cite
If you use these files or report results on this evaluation set, cite the paper that introduced the program and the
evaluation set, and the corpus:

```bibtex
@inproceedings{Sennrich2010MTbasedSA,
    title = {{MT-based Sentence Alignment for OCR-generated Parallel Texts}},
    author = "Rico Sennrich and Martin Volk",
    booktitle = "Proceedings of the 9th Conference of the Association for Machine Translation in the Americas: Research Papers",
    address = "Denver, Colorado",
    year = "2010",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2010.amta-papers.14",
}

@inproceedings{volk-etal-2010-challenges,
    title = "Challenges in Building a Multilingual Alpine Heritage Corpus",
    author = "Volk, Martin and Bubenhofer, Noah and Althaus, Adrian and Bangerter, Maya and Furrer, Lenz and Ruef, Beni",
    booktitle = "Proceedings of the Seventh International Conference on Language Resources and Evaluation (LREC'10)",
    year = "2010",
    address = "Valletta, Malta",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L10-1069/",
}
```

## Licence
Bleualign and its evaluation data are distributed under the **GNU General Public License, version 2**. These files
are output of that program on that data. To be on the safe side they are provided here under the same licence, the
GNU GPL version 2, and **not** under the Apache licence of TranslateAlign; they are test material only, are not
part of the installed package, and must not be included in a distribution of the package (wheel or source
distribution). The evaluation texts, the translations and the gold alignments themselves are not in this
repository; get them from the Bleualign repository.
