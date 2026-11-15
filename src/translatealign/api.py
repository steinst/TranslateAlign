"""Python interface.

    import translatealign
    pairs = translatealign.align(german, french, translation=german_in_french)          # a ready-made translation
    pairs = translatealign.align(german, french, model="opus-mt", source_language="de", target_language="fr")
    for p in pairs:
        print(p.src, p.trg, p.source, "|||", p.target)
    print(translatealign.cite(model="opus-mt"))                 # what to cite; bibtex=True for BibTeX

`german`, `french`: lists of sentences; a line ".EOA" in both separates documents. Numbers in the result are
indices into these lists.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

from . import citations
from .cli import SCORERS, align_files, read_lines, with_unaligned
from .search import DELIMITER

__all__ = ["Pair", "align", "cite", "read_lines"]


class Pair(NamedTuple):
    """Sentences `src` of the source text aligned with sentences `trg` of the target text (indices into the
    lists given to align(); one side is empty for an unaligned sentence)."""
    src: tuple
    trg: tuple
    source: str          # the source sentences, joined by a space
    target: str
    kind: Optional[str]  # "anchor" (first pass), "heuristic" (gap heuristics), "length" (Gale-Church); None if unaligned


def _lines(name: str, value) -> list:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a list of sentences, not a string (read a file with "
                        "translatealign.read_lines(path))")
    return list(value)


def _runs(name: str, value) -> list:
    """None, one translation (a list of strings) or several (a list of lists) -> a list of translations."""
    if value is None:
        return []
    value = _lines(name, value)
    if value and not isinstance(value[0], str):
        return [_lines(name, v) for v in value]
    return [value]


def align(source: Sequence[str], target: Sequence[str], translation=None, *, translation_of_target=None,
          model: Optional[str] = None, model_file: Optional[str] = None, source_language: Optional[str] = None,
          target_language: Optional[str] = None, translate: str = "source", scorer: str = "chrf",
          unaligned: bool = False, delimiter: str = DELIMITER, cache: Union[bool, str] = True,
          device: Optional[str] = None, trust_remote_code: bool = False, retry_failed: bool = False) -> list:
    """Align two texts (lists of sentences) and return a list of Pair in text order.

    translation:           the machine translation of `source` into the language of `target`, line by line
                           (or a list of several such translations).
    translation_of_target: the same for `target` (into the language of `source`).
    model / model_file:    translate here instead, with a model (a shipped name such as "opus-mt", a Hugging Face
                           id, or a definition file; needs `pip install "translatealign[models]"`, and
                           source_language / target_language). `translate`: "source", "target" or "both".
                           cache: True (the default cache file), a path, or False.
    Each translation is one run; with several, only the beads found in every run are kept.
    scorer:                "chrf" (chrF++ in both directions, default) or "bleu" (the settings of the papers).
    unaligned:             also return the sentences in no bead (one side empty, kind None).
    """
    src, trg = _lines("source", source), _lines("target", target)
    translations = _runs("translation", translation)
    translations_of_target = _runs("translation_of_target", translation_of_target)
    if scorer not in SCORERS:
        raise ValueError(f"scorer must be one of {', '.join(sorted(SCORERS))}")
    if model is not None or model_file is not None:
        if not (source_language and target_language):
            raise ValueError("a model needs source_language and target_language")
        if translate not in ("source", "target", "both"):
            raise ValueError("translate must be 'source', 'target' or 'both'")
        of_source, of_target = _translate(src, trg, model, model_file, source_language, target_language,
                                          translate, cache, device, trust_remote_code, retry_failed, delimiter)
        translations += of_source
        translations_of_target += of_target
    beads = align_files(src, trg, translations, translations_of_target, scorer=scorer, delimiter=delimiter)
    kinds = {(tuple(b.src), tuple(b.trg)): b.kind for b in beads}
    out = []
    for s, t in with_unaligned(beads, src, trg, delimiter):
        if not unaligned and not (s and t):
            continue
        out.append(Pair(s, t, " ".join(src[i] for i in s), " ".join(trg[j] for j in t), kinds.get((s, t))))
    return out


def _translate(src, trg, model, model_file, a, b, translate, cache, device, trust_remote_code, retry_failed,
               delimiter):
    from .translate import TranslationCache, load_translator
    translator = load_translator(model, model_file, device=device, trust_remote_code=trust_remote_code)
    store = None if cache is False or cache is None else TranslationCache(None if cache is True else cache)
    of_source, of_target = [], []
    try:
        if translate in ("source", "both"):
            of_source.append(list(translator.translate(src, a, b, cache=store, keep=[delimiter],
                                                       retry_failed=retry_failed)))
        if translate in ("target", "both"):
            of_target.append(list(translator.translate(trg, b, a, cache=store, keep=[delimiter],
                                                       retry_failed=retry_failed)))
    finally:
        translator.close()
        if store is not None:
            store.close()
    return of_source, of_target


def cite(scorer: str = "chrf", model: Optional[str] = None, model_file: Optional[str] = None,
         bibtex: bool = False) -> str:
    """What to cite for a run with this scorer and model (no model: a ready-made translation)."""
    with_model = model is not None or model_file is not None
    models = [citations.model_definition(model, model_file)] if with_model else []
    cites = citations.for_run(scorer, models, translation_files=not with_model)
    return citations.format_bibtex(cites) if bibtex else citations.format_text(cites)
