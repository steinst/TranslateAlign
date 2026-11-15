"""Translate the evaluation sets with one model; write the translations to WORK/<set>/<model>.{fwd,rev}.txt
(fwd = translation of the source side, rev = translation of the target side; lines as in the set, `.EOA` kept)
and WORK/<set>/<model>.json (seconds, counts, failed sentences, and the hashes of the source text and of the
translation written and the model's cache id: files are reused only if all three still match).

    python tools/translate_sets.py MODEL [--sets de-fr-dev de-fr-test en-is] [--directions fwd rev]
MODEL: a shipped name (opus-mt, nllb-200, madlad-400, translategemma-4b, ...).
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(__file__))
from evalsets import SETS, WORK, load  # noqa: E402

from translatealign.translate import TranslationCache, load_translator  # noqa: E402


def sha(lines) -> str:
    """A hash of a list of lines (the text that was translated, or the translation written)."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def reusable(path, lines, cache_id, retry_failed=False):
    """Can the files at `path` (.txt and .json) be reused as the translation of `lines` by the model with this
    cache id? Only if both exist, the json records the same hash of the source text and the same cache id (the
    model and every setting that changes its output) and the hash of the .txt that was written, and the .txt
    still has that hash. Files from before these fields existed are not reusable (they are translated again;
    the translation cache makes that quick). -> (True, "") or (False, reason)."""
    if not (os.path.exists(path + ".json") and os.path.exists(path + ".txt")):
        return False, "no files"
    try:
        with open(path + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        with open(path + ".txt", encoding="utf-8") as f:
            written = f.read().split("\n")[:-1]
    except (OSError, ValueError) as e:
        return False, f"unreadable ({e})"
    if "source_sha256" not in meta or "cache_id" not in meta or "translation_sha256" not in meta:
        return False, "made before the hashes were stored"
    if meta["source_sha256"] != sha(lines):
        return False, "the source text changed"
    if meta["cache_id"] != cache_id:
        return False, "the model or its definition changed"
    if meta["translation_sha256"] != sha(written) or len(written) != len(lines):
        return False, "the translation file was changed or damaged"
    if retry_failed and meta.get("failed"):
        return False, "had failed sentences and --retry-failed was given"
    return True, ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--sets", nargs="+", default=SETS)
    p.add_argument("--directions", nargs="+", default=["fwd", "rev"])
    p.add_argument("--retry-failed", action="store_true",
                   help="redo files that had failed sentences, trying those sentences again (after a change of the checks)")
    a = p.parse_args()
    translator = load_translator(a.model)
    cache = TranslationCache(os.path.join(WORK, "cache.sqlite"))
    for name in a.sets:
        s = load(name)
        os.makedirs(os.path.join(WORK, name), exist_ok=True)
        for d in a.directions:
            lines, x, y = (s.src, s.source_language, s.target_language) if d == "fwd" else \
                          (s.trg, s.target_language, s.source_language)
            path = os.path.join(WORK, name, f"{a.model}.{d}")
            try:
                ident = translator.cache_id(x, y)
            except Exception as e:     # a direction the model does not have (Lumen out of Icelandic)
                print(f"{name} {a.model} {d}: not possible: {e}", flush=True)
                continue
            ok, why = reusable(path, lines, ident, a.retry_failed)
            if ok:
                print("have", path, flush=True)
                continue
            if why != "no files":
                print(f"translating again, {path}: {why}", flush=True)
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    out = translator.translate(lines, x, y, cache=cache, keep=[".EOA"], retry_failed=a.retry_failed)
            except Exception as e:     # a direction the model does not have (Lumen out of Icelandic)
                print(f"{name} {a.model} {d}: not possible: {e}", flush=True)
                continue
            seconds = time.time() - t0
            with open(path + ".txt", "w", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in out)
            meta = dict(model=a.model, set=name, direction=d, languages=f"{x}-{y}", sentences=len(lines),
                        seconds=round(seconds, 1), translated=out.translated, from_cache=out.from_cache,
                        retried=out.retried, failed=len(out.problems), cache_id=ident,
                        source_sha256=sha(lines), translation_sha256=sha(list(out)),
                        failed_examples=[str(q)[:200] for q in out.problems[:5]])
            with open(path + ".json", "w") as f:
                json.dump(meta, f, indent=1)
            print(json.dumps(meta), flush=True)
    translator.close()
    cache.close()


if __name__ == "__main__":
    main()
