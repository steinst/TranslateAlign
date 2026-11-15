"""What to cite for a run: TranslateAlign, the two Bleualign papers, the similarity score, Gale-Church, the translation model.

    cites = for_run(scorer="chrf", models=[model_definition(model="opus-mt")])
    print(format_text(cites))       # one line per work (printed by the command line after each run)
    print(format_bibtex(cites))     # BibTeX (translatealign --bibtex)

A model definition file may carry its own `citation` (a BibTeX entry, or a line of text). Models without one are
cited by their Hugging Face model card.
"""
from __future__ import annotations

import re
from typing import Iterable, NamedTuple, Optional, Sequence

from . import __version__


class Citation(NamedTuple):
    key: str
    text: str         # one line, for the summary after a run
    bibtex: str       # one BibTeX entry ("" if there is none)


def _c(key: str, text: str, bibtex: str) -> Citation:
    return Citation(key, text, bibtex.strip())


REPOSITORY = "https://github.com/steinst/TranslateAlign"
TRANSLATEALIGN = _c(
    "steingrimsson-translatealign-2026",
    f"Steinthor Steingrimsson (2026). TranslateAlign (version {__version__}) [Software]. "
    f"See GitHub repo steinst/TranslateAlign: {REPOSITORY}",
    f"""@software{{steingrimsson-translatealign-2026,
  author = {{Steingrimsson, Steinthor}},
  title = {{TranslateAlign}},
  year = {{2026}},
  version = {{{__version__}}},
  url = {{{REPOSITORY}}}
}}""",
)

SENNRICH_2010 = _c("sennrich-volk-2010", "Rico Sennrich and Martin Volk (2010). MT-based Sentence Alignment for "
                   "OCR-generated Parallel Texts. AMTA 2010.", """
@inproceedings{sennrich-volk-2010,
  title = {{MT}-based Sentence Alignment for {OCR}-generated Parallel Texts},
  author = {Sennrich, Rico and Volk, Martin},
  booktitle = {Proceedings of the Ninth Conference of the Association for Machine Translation in the Americas (AMTA 2010)},
  address = {Denver, Colorado, USA},
  year = {2010}
}""")

SENNRICH_2011 = _c("sennrich-volk-2011", "Rico Sennrich and Martin Volk (2011). Iterative, MT-based Sentence "
                   "Alignment of Parallel Texts. NODALIDA 2011, pp. 175-182.", """
@inproceedings{sennrich-volk-2011,
  title = {Iterative, {MT}-based Sentence Alignment of Parallel Texts},
  author = {Sennrich, Rico and Volk, Martin},
  booktitle = {Proceedings of the 18th Nordic Conference of Computational Linguistics (NODALIDA 2011)},
  pages = {175--182},
  address = {Riga, Latvia},
  year = {2011}
}""")

GALE_CHURCH = _c("gale-church-1993", "William A. Gale and Kenneth W. Church (1993). A Program for Aligning "
                 "Sentences in Bilingual Corpora. Computational Linguistics 19(1), pp. 75-102.", """
@article{gale-church-1993,
  title = {A Program for Aligning Sentences in Bilingual Corpora},
  author = {Gale, William A. and Church, Kenneth W.},
  journal = {Computational Linguistics},
  volume = {19},
  number = {1},
  pages = {75--102},
  year = {1993}
}""")

BLEU = _c("papineni-etal-2002", "Kishore Papineni, Salim Roukos, Todd Ward and Wei-Jing Zhu (2002). BLEU: a Method "
          "for Automatic Evaluation of Machine Translation. ACL 2002, pp. 311-318.", """
@inproceedings{papineni-etal-2002,
  title = {{BLEU}: a Method for Automatic Evaluation of Machine Translation},
  author = {Papineni, Kishore and Roukos, Salim and Ward, Todd and Zhu, Wei-Jing},
  booktitle = {Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics},
  pages = {311--318},
  address = {Philadelphia, Pennsylvania, USA},
  year = {2002}
}""")

CHRF = _c("popovic-2015", "Maja Popović (2015). chrF: character n-gram F-score for automatic MT evaluation. "
          "WMT 2015, pp. 392-395.", """
@inproceedings{popovic-2015,
  title = {{chrF}: character n-gram {F}-score for automatic {MT} evaluation},
  author = {Popovi{\\'c}, Maja},
  booktitle = {Proceedings of the Tenth Workshop on Statistical Machine Translation},
  pages = {392--395},
  address = {Lisbon, Portugal},
  year = {2015}
}""")

CHRF_PP = _c("popovic-2017", "Maja Popović (2017). chrF++: words helping character n-grams. WMT 2017, "
             "pp. 612-618.", """
@inproceedings{popovic-2017,
  title = {{chrF++}: words helping character n-grams},
  author = {Popovi{\\'c}, Maja},
  booktitle = {Proceedings of the Second Conference on Machine Translation},
  pages = {612--618},
  address = {Copenhagen, Denmark},
  year = {2017}
}""")

OPUS_MT = _c("tiedemann-thottingal-2020", "Jörg Tiedemann and Santhosh Thottingal (2020). OPUS-MT - Building open "
             "translation services for the World. EAMT 2020, pp. 479-480.", """
@inproceedings{tiedemann-thottingal-2020,
  title = {{OPUS-MT} -- Building open translation services for the World},
  author = {Tiedemann, J{\\"o}rg and Thottingal, Santhosh},
  booktitle = {Proceedings of the 22nd Annual Conference of the European Association for Machine Translation},
  pages = {479--480},
  address = {Lisboa, Portugal},
  year = {2020}
}""")

NLLB = _c("nllb-2022", "NLLB Team et al. (2022). No Language Left Behind: Scaling Human-Centered Machine "
          "Translation. arXiv:2207.04672.", """
@misc{nllb-2022,
  title = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  author = {{NLLB Team} and Costa-juss{\\`a}, Marta R. and others},
  year = {2022},
  eprint = {2207.04672},
  archivePrefix = {arXiv}
}""")

MADLAD = _c("kudugunta-etal-2023", "Sneha Kudugunta et al. (2023). MADLAD-400: A Multilingual And Document-Level "
            "Large Audited Dataset. arXiv:2309.04662.", """
@misc{kudugunta-etal-2023,
  title = {{MADLAD-400}: A Multilingual And Document-Level Large Audited Dataset},
  author = {Kudugunta, Sneha and Caswell, Isaac and Zhang, Biao and Garcia, Xavier and Choquette-Choo, Christopher A. and Lee, Katherine and Xin, Derrick and Kusupati, Aditya and Stella, Romi and Bapna, Ankur and Firat, Orhan},
  year = {2023},
  eprint = {2309.04662},
  archivePrefix = {arXiv}
}""")

FAMILIES = {"opus-mt": OPUS_MT, "nllb": NLLB, "madlad": MADLAD}
SCORERS = {"bleu": [BLEU], "chrf": [CHRF, CHRF_PP]}
TRANSLATION_FILES = Citation("translation-files", "the system that made the translation file(s) you supplied", "")


def model_definition(model: Optional[str] = None, model_file: Optional[str] = None):
    """The definition of a model as --model / --model-file name it, without loading anything."""
    from .translate import ModelDefinition, load_definition, shipped_definition
    from .translate.languages import family_of
    if model_file is not None:
        return load_definition(model_file)
    shipped = shipped_definition(model)
    if shipped:
        return load_definition(shipped)
    return ModelDefinition(name=model, model=model, backend="seq2seq", family=family_of(model))


def for_model(definition) -> Citation:
    """A model definition -> its citation: the definition's own, else its family's paper, else its model card."""
    own = (definition.citation or "").strip()
    if own.startswith("@"):
        key = re.match(r"@\w+\s*\{\s*([^,\s]+)", own)
        return Citation(key.group(1) if key else definition.name, _bibtex_title(own) or definition.name, own)
    if own:
        return Citation(definition.name, own, "")
    if definition.family in FAMILIES:
        return FAMILIES[definition.family]
    model = definition.model
    if "/" in model and not model.startswith(("/", ".", "~")) and "{" not in model:
        url = f"https://huggingface.co/{model}"
        key = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()
        return Citation(key, f"the model card of {model}: {url}", f"""@misc{{{key},
  title = {{{model}}},
  howpublished = {{\\url{{{url}}}}},
  note = {{Model card on Hugging Face}}
}}""")
    return Citation(definition.name, f"the translation model {model}", "")


def _bibtex_title(entry: str) -> str:
    m = re.search(r"title\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", entry)
    return m.group(1).replace("{", "").replace("}", "") if m else ""


def for_run(scorer: str = "chrf", models: Iterable = (), translation_files: bool = False) -> list:
    """Everything a run used: the papers of the search, Gale-Church, the scorer, each model (definitions)."""
    out = [TRANSLATEALIGN, SENNRICH_2010, SENNRICH_2011, GALE_CHURCH] + SCORERS[scorer]
    for d in models:
        out.append(for_model(d))
    if translation_files:
        out.append(TRANSLATION_FILES)
    seen, unique = set(), []
    for c in out:
        if c.key not in seen:
            seen.add(c.key)
            unique.append(c)
    return unique


def format_text(citations: Sequence[Citation]) -> str:
    return "\n".join("  " + c.text for c in citations)


def format_bibtex(citations: Sequence[Citation]) -> str:
    entries = [c.bibtex for c in citations if c.bibtex]
    return "\n\n".join(entries) + "\n" if entries else ""
