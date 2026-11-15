"""Align the evaluation sets from the translations made by translate_sets.py; per-document counts to JSON.

    python tools/evaluate_all.py --sets de-fr-dev --configs bleu chrf --out WORK/dev.json
    python tools/evaluate_all.py --sets de-fr-test en-is --configs bleu chrf --out WORK/test.json

A config is a scorer plus search options, written `scorer[:opt=value,...]`; options: k, max_items, max_sentences,
source_more, target_more (matches conditions), one_to_one, beta (chrf), word (chrf word order), both (chrf both
directions, default 1). Result key: `set|model|config|mode` with mode fwd, rev or both (intersection); value: the
counts of translatealign.evaluate.counts per document (a list), so that the report can bootstrap over documents.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from evalsets import WORK, check_pairs, load  # noqa: E402

from translatealign import cli  # noqa: E402
from translatealign.evaluate import counts  # noqa: E402
from translatealign.scoring import BleuScorer, ChrfScorer  # noqa: E402
from translatealign.search import GapOptions, flip, intersect  # noqa: E402
from translatealign.search import align_documents  # noqa: E402


def make(config):
    name, _, rest = config.partition(":")
    opts = dict(kv.split("=") for kv in rest.split(",")) if rest else {}
    gap = {}
    for key, field, conv in (("max_items", "max_items", int), ("max_sentences", "max_sentences", int),
                             ("source_more", "source_needs_more_matches", lambda v: bool(int(v))),
                             ("target_more", "target_needs_more_matches", lambda v: bool(int(v))),
                             ("one_to_one", "one_to_one", lambda v: bool(int(v)))):
        if key in opts:
            gap[field] = conv(opts[key])
    kwargs = {"k": int(opts.get("k", 3))}
    if gap:
        kwargs["options"] = GapOptions(**gap)
    if name == "bleu":
        scorer = BleuScorer.reference()
    elif name == "chrf":
        scorer = ChrfScorer(beta=float(opts.get("beta", 2)), word_order=int(opts.get("word", 2)),
                            both_directions=bool(int(opts.get("both", 1))))
    else:
        raise ValueError(config)
    return scorer, kwargs


def run(s, fwd, rev, config):
    """-> {mode: per-document counts}; fwd / rev are translations (or None)."""
    scorer, kwargs = make(config)
    runs = {}
    if fwd is not None:
        runs["fwd"] = align_documents(fwd, s.trg, scorer, source=s.src, **kwargs)
    if rev is not None:
        runs["rev"] = flip(align_documents(rev, s.src, scorer, source=s.trg, **kwargs))
    if len(runs) == 2:
        runs["both"] = intersect(runs["fwd"], runs["rev"])
    out = {}
    for mode, beads in runs.items():
        complete = cli.with_unaligned(beads, s.src, s.trg)
        docs = cli.per_document(complete, s.src, s.trg)
        out[mode] = [counts(g, t) for g, t in check_pairs(s.gold, docs, s.name)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sets", nargs="+", required=True)
    p.add_argument("--configs", nargs="+", default=["bleu", "chrf"])
    p.add_argument("--models", nargs="*", help="default: every model with translations for the set")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name in a.sets:
        s = load(name)
        models = a.models or sorted({os.path.basename(f).split(".")[0] for f in glob.glob(f"{WORK}/{name}/*.json")})
        for model in models:
            def read(d):
                path = f"{WORK}/{name}/{model}.{d}.txt"
                if not os.path.exists(path):
                    return None
                return cli.read_lines(path)
            fwd, rev = read("fwd"), read("rev")
            for config in a.configs:
                t0 = time.time()
                for mode, c in run(s, fwd, rev, config).items():
                    results[f"{name}|{model}|{config}|{mode}"] = c
                print(name, model, config, f"{time.time() - t0:.1f}s", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
