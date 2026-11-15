"""Command line: align two texts with a translation of one of them (or of both), ready-made or made here.

    translatealign SRC TRG --translation FILE [--translation-of-target FILE] -o OUT
    translatealign SRC TRG --model MODEL -s de -t fr [--translate source|target|both] -o OUT
    translatealign --bibtex [--scorer bleu] [--model MODEL]       (what to cite, as BibTeX)

SRC, TRG: one sentence per line; lines `.EOA` are hard delimiters between documents (the same number in both).
--translation: the machine translation of SRC into the language of TRG, line by line.
--translation-of-target: the machine translation of TRG into the language of SRC, line by line.
--model / --model-file: translate with a model (translatealign.translate); the translations are cached.
Each translation is one run; with several, only the beads found in every run are kept.

Written: OUT.path (one bead per line, `[0, 1]:[2]`, line numbers counted from 0), OUT.src and OUT.trg (the
aligned sentences, line by line; the sentences of a bead joined by a space).
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import warnings
from bisect import bisect_right
from typing import Optional, Sequence

from . import __version__, citations
from .evaluate import format_bead, format_scores, read_alignment, score
from .scoring import BleuScorer, ChrfScorer
from .translate.cache import ENV as CACHE_ENV
from .search import ANCHOR, DELIMITER, HEURISTIC, LENGTH, Bead, align_documents, flip, intersect, split_documents

# bleu: the settings of the original. chrf: chrF++ in both directions, harmonic mean, as
# the papers do for BLEU; chosen on the development document (docs/RESULTS.md): plain chrF++ (beta = 2, the
# translation as hypothesis) favours recall and joins too many sentences.
SCORERS = {"bleu": BleuScorer.reference, "chrf": lambda: ChrfScorer(both_directions=True)}


def read_lines(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def align_files(src: Sequence[str], trg: Sequence[str], translations: Sequence[Sequence[str]] = (),
                translations_of_target: Sequence[Sequence[str]] = (), scorer="chrf",
                delimiter: str = DELIMITER) -> list:
    """Align the lines of two texts. One run per translation; several runs are intersected.

    translations: translations of `src` (aligned with `trg`); translations_of_target: translations of `trg`
    (aligned with `src`, then turned round). Returns beads whose numbers are line numbers of `src` and `trg`.
    """
    if not translations and not translations_of_target:
        raise ValueError("at least one translation is needed")
    if isinstance(scorer, str):
        scorer = SCORERS[scorer]()
    runs = []
    for name, translated, own, other, turn in (
            [("translation", x, src, trg, False) for x in translations]
            + [("translation of the target", x, trg, src, True) for x in translations_of_target]):
        if len(translated) != len(own):
            raise ValueError(f"the {name} has {len(translated)} lines, the text it translates {len(own)}")
        beads = align_documents(translated, other, scorer, source=own, delimiter=delimiter)
        runs.append(flip(beads) if turn else beads)
    return runs[0] if len(runs) == 1 else intersect(*runs)


def with_unaligned(beads: Sequence[Bead], src: Sequence[str], trg: Sequence[str],
                   delimiter: str = DELIMITER) -> list:
    """The beads as (source numbers, target numbers) in text order, with every sentence that is in no bead
    added as a bead with one empty side (delimiter lines are left out)."""
    used_s = {i for b in beads for i in b.src}
    used_t = {j for b in beads for j in b.trg}
    out, s, t = [], 0, 0

    def flush(s_end, t_end):
        nonlocal s, t
        for i in range(s, s_end):
            if i not in used_s and src[i].strip() != delimiter:
                out.append(((i,), ()))
        for j in range(t, t_end):
            if j not in used_t and trg[j].strip() != delimiter:
                out.append(((), (j,)))
        s, t = max(s, s_end), max(t, t_end)

    for b in beads:
        flush(b.src[0], b.trg[0])
        out.append((tuple(b.src), tuple(b.trg)))
    flush(len(src), len(trg))
    return out


def per_document(beads: Sequence, src: Sequence[str], trg: Sequence[str], delimiter: str = DELIMITER) -> list:
    """Split beads (line numbers of the files) into one list per document, numbers counted from 0 within the
    document: the form of gold alignments kept per document."""
    s_starts = [start for start, _ in split_documents(src, delimiter)]
    t_starts = [start for start, _ in split_documents(trg, delimiter)]
    docs = [[] for _ in s_starts]
    for b in beads:
        s, t = tuple(b[0]), tuple(b[1])
        d = bisect_right(s_starts, s[0]) - 1 if s else bisect_right(t_starts, t[0]) - 1
        docs[d].append((tuple(i - s_starts[d] for i in s), tuple(j - t_starts[d] for j in t)))
    return docs


def _norm(path: str) -> str:
    """Resolved path (symbolic links followed); case-folded on Windows and macOS, whose file systems usually
    ignore case."""
    p = os.path.realpath(path)
    return os.path.normcase(p.lower() if sys.platform in ("win32", "darwin") else p)


def find_collision(outputs: Sequence[str], inputs: Sequence[str]):
    """The first (output, input) pair that names the same file, or None. Compares resolved paths and, where
    both files exist, `os.path.samefile` (which also catches hard links)."""
    for out in outputs:
        for inp in inputs:
            if _norm(out) == _norm(inp):
                return out, inp
            try:
                if os.path.exists(out) and os.path.exists(inp) and os.path.samefile(out, inp):
                    return out, inp
            except OSError:
                pass
    return None


def output_paths(prefix: str, save_translations: bool = False) -> list:
    """Every file a run writes: OUT.path, OUT.src, OUT.trg and, with --save-translations, OUT.translation and
    OUT.translation-of-target."""
    suffixes = [".path", ".src", ".trg"] + ([".translation", ".translation-of-target"] if save_translations else [])
    return [prefix + x for x in suffixes]


def input_paths(args) -> list:
    """Every file a run reads: SRC, TRG, the translations, the gold files, the model file and the cache file."""
    paths = [args.src, args.trg, *args.translation, *args.translation_of_target, *(args.gold or [])]
    if args.model_file:
        paths.append(args.model_file)
    if (args.model or args.model_file) and not args.no_cache:
        from .translate.cache import default_cache_path
        paths.append(args.cache or default_cache_path())
    return [p for p in paths if p]


def check_outputs(args) -> None:
    """Raise ValueError naming both paths if an output file is an input file (before anything is translated)."""
    hit = find_collision(output_paths(args.output, args.save_translations), input_paths(args))
    if hit:
        raise ValueError(f"the output file {hit[0]} is the same file as the input file {hit[1]}; "
                         "choose another output prefix (-o), nothing was changed")


@contextlib.contextmanager
def atomic_writer(path: str):
    """Open a temporary file next to `path`; on leaving the block normally it replaces `path` (os.replace),
    on an error it is removed and `path` is untouched."""
    directory = os.path.dirname(os.path.abspath(path))
    f = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory, prefix=".translatealign-",
                                    suffix=".tmp", delete=False)
    try:
        with f:
            yield f
        os.replace(f.name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(f.name)
        raise


def write_output(prefix: str, beads: Sequence, src: Sequence[str], trg: Sequence[str]) -> None:
    """beads: (source numbers, target numbers); writes prefix.path, prefix.src, prefix.trg.

    Each file is written completely to a temporary file first; only when all three are complete are they put
    in place, one after the other (the three are not replaced in one step: an interruption between the
    replacements can leave a new file next to an old one)."""
    names = [prefix + ".path", prefix + ".src", prefix + ".trg"]
    temporary = []
    try:
        for name in names:
            f = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=os.path.dirname(os.path.abspath(name)),
                                            prefix=".translatealign-", suffix=".tmp", delete=False)
            temporary.append(f)
        for s, t in beads:
            temporary[0].write(format_bead(s, t) + "\n")
            temporary[1].write(" ".join(src[i] for i in s) + "\n")
            temporary[2].write(" ".join(trg[j] for j in t) + "\n")
        for f in temporary:
            f.close()
        for f, name in zip(temporary, names):
            os.replace(f.name, name)
    except BaseException:
        for f in temporary:
            f.close()
            with contextlib.suppress(OSError):
                os.unlink(f.name)
        raise


def translate_files(args, src: Sequence[str], trg: Sequence[str]):
    """Translate with the model of --model / --model-file. -> (translations of SRC, translations of TRG),
    each a list with no or one translation."""
    from .translate import TranslationCache, describe_problems, load_translator
    translator = load_translator(args.model, args.model_file, device=args.device,
                                 trust_remote_code=args.trust_remote_code)
    cache = None if args.no_cache else TranslationCache(args.cache)
    jobs = []
    if args.translate in ("source", "both"):
        jobs.append(("source", src, args.source_language, args.target_language, ".translation"))
    if args.translate in ("target", "both"):
        jobs.append(("target", trg, args.target_language, args.source_language, ".translation-of-target"))
    results = {"source": [], "target": []}
    try:
        for side, lines, a, b, suffix in jobs:
            def progress(done, total, side=side, a=a, b=b):
                if not args.quiet and sys.stderr.isatty():
                    print(f"\rtranslating the {side} text {a} -> {b}: {done}/{total}", end="", file=sys.stderr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)     # reported below, with examples
                out = translator.translate(lines, a, b, cache=cache, keep=[args.delimiter],
                                           retry_failed=args.retry_failed, progress=progress)
            if not args.quiet:
                if sys.stderr.isatty() and out.translated:
                    print(file=sys.stderr)
                print(f"{side} text {a} -> {b} with {translator.name}: {out.translated} sentences translated "
                      f"({out.retried} needed a second attempt), {out.from_cache} from the cache"
                      + (f", {out.known_failures} known from the cache to fail (--retry-failed tries again)"
                         if out.known_failures else "")
                      + (f"; {out.rechecked} cached translations failed the current checks and were translated "
                         "again" if out.rechecked else "")
                      + (f"; {len(out.truncated)} translations stopped at max_new_tokens and may be cut off"
                         if out.truncated else ""), file=sys.stderr)
            if out.truncated:
                print(f"translatealign: warning: {len(out.truncated)} translation(s) of the {side} text stopped at "
                      f"the limit on new tokens and may be cut off (first at line {out.truncated[0][0]}): raise "
                      f"max_new_tokens in the definition's [generation] table, or split the sentence. They are not "
                      f"stored in the cache.", file=sys.stderr)
            if out.problems:
                print(f"translatealign: warning: {len(out.problems)} sentence(s) of the {side} text could not "
                      f"be translated and are aligned by length only:\n{describe_problems(out.problems)}",
                      file=sys.stderr)
            if args.save_translations:
                with atomic_writer(args.output + suffix) as f:
                    f.writelines(line + "\n" for line in out)
            results[side].append(list(out))
    finally:
        translator.close()
        if cache is not None:
            cache.close()
    return results["source"], results["target"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="translatealign",
        description="Sentence alignment with a machine translation of one side (the search of the Bleualign "
                    "papers, re-implemented; not the official Bleualign).")
    p.add_argument("src", metavar="SRC", nargs="?", help="source text, one sentence per line")
    p.add_argument("trg", metavar="TRG", nargs="?", help="target text, one sentence per line")
    p.add_argument("--translation", metavar="FILE", action="append", default=[],
                   help="translation of SRC into the language of TRG, line by line (can be repeated)")
    p.add_argument("--translation-of-target", metavar="FILE", action="append", default=[],
                   help="translation of TRG into the language of SRC, line by line (can be repeated); with "
                        "more than one translation only the beads found in every run are kept")
    m = p.add_argument_group("translating with a model (needs: pip install \"translatealign[models]\")")
    m.add_argument("--model", metavar="ID",
                   help="Hugging Face id or directory of an Opus-MT, NLLB-200 or MADLAD-400 model; {src} and "
                        "{trg} in the id stand for the language codes (Helsinki-NLP/opus-mt-{src}-{trg})")
    m.add_argument("--model-file", metavar="FILE", help="a model definition file (TOML, see docs/MODELS.md)")
    m.add_argument("-s", "--source-language", metavar="CODE", help="language of SRC (de, fr, is, ...)")
    m.add_argument("-t", "--target-language", metavar="CODE", help="language of TRG")
    m.add_argument("--translate", choices=["source", "target", "both"], default="source",
                   help="which text the model translates: SRC into the language of TRG (default), TRG into "
                        "the language of SRC, or both (two runs, intersected)")
    m.add_argument("--cache", metavar="FILE", help=f"translation cache (default: ${CACHE_ENV}, else "
                                                   "translatealign/translations.sqlite in the user's cache directory)")
    m.add_argument("--no-cache", action="store_true", help="do not read or write the translation cache")
    m.add_argument("--retry-failed", action="store_true",
                   help="translate again the sentences that the cache remembers as failed")
    m.add_argument("--device", metavar="DEVICE", help="cuda, cuda:1, cpu, mps (default: a GPU if there is one)")
    m.add_argument("--trust-remote-code", action="store_true",
                   help="allow a model that runs code downloaded from its repository (read that code first)")
    m.add_argument("--save-translations", action="store_true",
                   help="write the model's translations to OUT.translation / OUT.translation-of-target")
    p.add_argument("-o", "--output", metavar="OUT",
                   help="output prefix: OUT.path (beads), OUT.src and OUT.trg (aligned sentences); refused if one "
                        "of them is an input file. Each file is written to a temporary file and then put in "
                        "place; the three are replaced one after the other, not in one step")
    p.add_argument("--scorer", choices=sorted(SCORERS), default="chrf",
                   help="similarity of translation and target: chrf (chrF++ in both directions; default, "
                        "see docs/RESULTS.md) or bleu (the settings of the papers)")
    p.add_argument("--unaligned", action="store_true",
                   help="also write the sentences that are in no bead (a bead with one empty side)")
    p.add_argument("--delimiter", default=DELIMITER,
                   help="line that separates documents; the search never crosses it (default: %(default)s)")
    p.add_argument("--gold", metavar="FILE", nargs="+",
                   help="gold alignment, one file per document in text order (numbers counted from 0 within "
                        "the document): print strict and lax precision, recall and F1")
    p.add_argument("--bibtex", action="store_true",
                   help="print BibTeX for what the run used (the papers, the scorer, Gale-Church, the model) "
                        "on standard output; without SRC and TRG: print it and stop")
    p.add_argument("-q", "--quiet", action="store_true", help="no summary and no citations on standard error")
    p.add_argument("--version", action="version", version="translatealign " + __version__)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    with_model = bool(args.model or args.model_file)
    if args.model and args.model_file:
        parser.error("give --model or --model-file, not both")
    try:
        cites = citations.for_run(args.scorer, [citations.model_definition(args.model, args.model_file)]
                                  if with_model else [], translation_files=bool(args.translation
                                                                                or args.translation_of_target))
    except (OSError, ValueError) as e:
        print(f"translatealign: error: {e}", file=sys.stderr)
        return 2
    if args.bibtex and args.src is None and args.trg is None:
        sys.stdout.write(citations.format_bibtex(cites))
        return 0
    if args.src is None or args.trg is None:
        parser.error("give SRC and TRG")
    if args.output is None:
        parser.error("give the output prefix: -o OUT")
    if with_model and not (args.source_language and args.target_language):
        parser.error("--model / --model-file need --source-language and --target-language")
    if not with_model and not args.translation and not args.translation_of_target:
        parser.error("give --translation FILE and/or --translation-of-target FILE, or a model (--model, "
                     "--model-file)")
    try:
        check_outputs(args)
        src, trg = read_lines(args.src), read_lines(args.trg)
        translations = [read_lines(f) for f in args.translation]
        translations_of_target = [read_lines(f) for f in args.translation_of_target]
        if with_model:
            of_source, of_target = translate_files(args, src, trg)
            translations += of_source
            translations_of_target += of_target
        beads = align_files(src, trg, translations, translations_of_target,
                            scorer=args.scorer, delimiter=args.delimiter)
        gold = [read_alignment(f) for f in args.gold] if args.gold else None
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        print(f"translatealign: error: {e}", file=sys.stderr)
        return 2
    complete = with_unaligned(beads, src, trg, args.delimiter)
    write_output(args.output, complete if args.unaligned else [b for b in complete if b[0] and b[1]], src, trg)
    if not args.quiet:
        n_docs = len(split_documents(src, args.delimiter))
        kinds = {k: sum(b.kind == k for b in beads) for k in (ANCHOR, HEURISTIC, LENGTH)}
        runs = len(translations) + len(translations_of_target)
        print(f"{n_docs} document(s), {len(src) - n_docs + 1} / {len(trg) - n_docs + 1} sentences, scorer "
              f"{args.scorer}, {runs} run(s){' intersected' if runs > 1 else ''}\n"
              f"{len(beads)} beads: {kinds[ANCHOR]} first pass, {kinds[HEURISTIC]} gap heuristics, "
              f"{kinds[LENGTH]} Gale-Church; not aligned: {sum(not b[1] for b in complete)} source, "
              f"{sum(not b[0] for b in complete)} target sentences\n"
              f"written: {args.output}.path, {args.output}.src, {args.output}.trg\n"
              f"Please cite (BibTeX: add --bibtex):\n{citations.format_text(cites)}", file=sys.stderr)
    if gold is not None:
        try:
            print(format_scores(score(gold, per_document(complete, src, trg, args.delimiter))))
        except ValueError as e:
            print(f"translatealign: error: {e}", file=sys.stderr)
            return 2
    if args.bibtex:
        sys.stdout.write(citations.format_bibtex(cites))
    return 0


if __name__ == "__main__":
    sys.exit(main())
