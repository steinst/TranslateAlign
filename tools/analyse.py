"""Error analysis: where TranslateAlign loses against SentAlign 2, and whether the translation model matters.

    python tools/analyse.py

1. Recall per gold bead shape (1-1, 1-2, 2-1, ...) and precision per search stage (first pass, gap heuristics,
   Gale-Church), for chrF++ in one and both directions, against SentAlign 2's output.
2. Translation quality (corpus chrF++ of each translation against the target sentences of the gold 1-1 beads)
   next to alignment strict F1.
3. Paired bootstrap of the differences between translation models (chrF++, both directions).
4. What the empty translations (failed checks) cost: those sentences and the beads that contain them.
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import sacrebleu

sys.path.insert(0, os.path.dirname(__file__))
from evalsets import WORK, check_pairs, load  # noqa: E402
from evaluate_all import make  # noqa: E402
from report import f1s, load as load_counts, resamples  # noqa: E402
from sentalign_baseline import ROOT, SETS as SA_SETS  # noqa: E402

from translatealign import cli  # noqa: E402
from translatealign.evaluate import read_alignment  # noqa: E402
from translatealign.search import align_documents, flip, intersect, split_documents  # noqa: E402

MODELS = ["opus-mt", "nllb-200", "madlad-400", "translategemma-4b", "translategemma-12b", "qwen-mt-lumen"]


def shape(b):
    return f"{len(b[0])}-{len(b[1])}"


def translations(name, model):
    out = {}
    for d in ("fwd", "rev"):
        p = f"{WORK}/{name}/{model}.{d}.txt"
        if os.path.exists(p):
            out[d] = cli.read_lines(p)
    return out


def runs(s, tr, config="chrf"):
    """-> {mode: per-document list of (src, trg, kind)} in document numbering."""
    scorer, kw = make(config)
    beads = {}
    if "fwd" in tr:
        beads["fwd"] = align_documents(tr["fwd"], s.trg, scorer, source=s.src, **kw)
    if "rev" in tr:
        beads["rev"] = flip(align_documents(tr["rev"], s.src, scorer, source=s.trg, **kw))
    if len(beads) == 2:
        beads["both"] = intersect(beads["fwd"], beads["rev"])
    s_starts = [a for a, _ in split_documents(s.src)]
    t_starts = [a for a, _ in split_documents(s.trg)]
    out = {}
    for mode, bs in beads.items():
        docs = [[] for _ in s_starts]
        for b in bs:
            if not (b.src and b.trg):
                continue
            d = np.searchsorted(s_starts, b.src[0], side="right") - 1
            docs[d].append((tuple(i - s_starts[d] for i in b.src), tuple(j - t_starts[d] for j in b.trg), b.kind))
        out[mode] = docs
    return out


def sentalign(name, s):
    folder, docs = SA_SETS[name]
    docs = docs or s.documents
    return [[(tuple(a), tuple(b), "sentalign") for a, b in read_alignment(f"{ROOT}/v2/{folder}/output/{d}.txt.path")
             if a and b] for d in docs]


def by_shape(gold, test, name):
    """recall per gold shape, precision per test shape."""
    rec, prec = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for g, t in check_pairs(gold, test, name):
        gs = {(a, b) for a, b in g if a and b}
        ts = {(a, b) for a, b, _ in t}
        for b in gs:
            rec[shape(b)][0] += b in ts
            rec[shape(b)][1] += 1
        for b in ts:
            prec[shape(b)][0] += b in gs
            prec[shape(b)][1] += 1
    return rec, prec


def by_kind(gold, test, name):
    out = defaultdict(lambda: [0, 0])
    for g, t in check_pairs(gold, test, name):
        gs = {(a, b) for a, b in g if a and b}
        for a, b, k in t:
            out[k][0] += (a, b) in gs
            out[k][1] += 1
    return out


def fmt(c):
    return f"{c[0] / c[1]:.3f} ({c[0]}/{c[1]})" if c[1] else "-"


def section1(name, s):
    print(f"\n#### {name}: recall per gold bead shape (strict)\n")
    systems = {"SentAlign 2": sentalign(name, s)}
    for m in ("opus-mt", "madlad-400"):
        r = runs(s, translations(name, m))
        systems[f"{m} source"] = r["fwd"]
        systems[f"{m} both"] = r["both"]
    shapes = Counter(shape(b) for g in s.gold for b in g if b[0] and b[1])
    common = [sh for sh, _ in shapes.most_common() if shapes[sh] >= 3]
    print("| system | " + " | ".join(f"{sh} ({shapes[sh]})" for sh in common) + " | other |")
    print("|---|" + "---|" * (len(common) + 1))
    for label, test in systems.items():
        rec, _ = by_shape(s.gold, test, name)
        other = [sum(rec[k][i] for k in rec if k not in common) for i in (0, 1)]
        print(f"| {label} | " + " | ".join(fmt(rec[sh]) for sh in common) + f" | {fmt(other)} |")
    print(f"\n#### {name}: precision per bead shape of the output, and per search stage\n")
    print("| system | 1-1 | 1-2 / 2-1 | larger | first pass | gap heuristics | Gale-Church |")
    print("|---|---|---|---|---|---|---|")
    for label, test in systems.items():
        _, prec = by_shape(s.gold, test, name)
        two = [prec["1-2"][i] + prec["2-1"][i] for i in (0, 1)]
        big = [sum(prec[k][i] for k in prec if k not in ("1-1", "1-2", "2-1")) for i in (0, 1)]
        kinds = by_kind(s.gold, test, name)
        print(f"| {label} | {fmt(prec['1-1'])} | {fmt(two)} | {fmt(big)} | "
              + " | ".join(fmt(kinds[k]) if k in kinds else "-" for k in ("anchor", "heuristic", "length")) + " |")


def section1b(name, s):
    """Gold beads that SentAlign finds and opus-mt (both) misses: what TranslateAlign did with them."""
    sa = sentalign(name, s)
    ta = runs(s, translations(name, "opus-mt"))["both"]
    fate = Counter()
    check_pairs(s.gold, sa, name, "SentAlign 2 documents")
    check_pairs(s.gold, ta, name, "TranslateAlign documents")
    for g, a, t in zip(s.gold, sa, ta):
        ga = {(x, y) for x, y in g if x and y}
        sa_set = {(x, y) for x, y, _ in a}
        ta_set = {(x, y) for x, y, _ in t}
        src_in = {i: (x, y) for x, y, _ in t for i in x}
        for b in ga & sa_set - ta_set:
            covering = {src_in.get(i) for i in b[0]}
            if covering == {None}:
                fate["source sentence(s) left unaligned (dropped by the intersection or a gap)"] += 1
            elif len(covering - {None}) == 1 and None not in covering:
                (x, y), = covering
                fate["merged into a larger bead" if set(b[0]) <= set(x) and set(b[1]) <= set(y) and (x, y) != b
                     else "split or aligned elsewhere"] += 1
            else:
                fate["split or aligned elsewhere"] += 1
        fate["(found by TranslateAlign, missed by SentAlign)"] += len(ga & ta_set - sa_set)
    print(f"\n#### {name}: gold beads SentAlign 2 finds and opus-mt / chrF++ / both misses\n")
    for k, v in fate.most_common():
        print(f"* {k}: {v}")


def section2(name, s):
    print(f"\n#### {name}: translation quality (corpus chrF++ against the gold 1-1 target sentences) and strict F1\n")
    counts = load_counts([f"{WORK}/all.json"])
    s_starts = [a for a, _ in split_documents(s.src)]
    t_starts = [a for a, _ in split_documents(s.trg)]
    pairs_f = [(s_starts[d] + b[0][0], t_starts[d] + b[1][0]) for d, g in enumerate(s.gold) for b in g
               if len(b[0]) == 1 and len(b[1]) == 1]
    print("| model | chrF++ of source -> target | of target -> source | F1 source | F1 target | F1 both |")
    print("|---|---|---|---|---|---|")
    for m in MODELS:
        tr = translations(name, m)
        if not tr:
            continue
        q = {}
        for d, (hyp_side, ref_side) in (("fwd", (0, 1)), ("rev", (1, 0))):
            if d in tr:
                refs = [(s.trg if ref_side else s.src)[p[ref_side]] for p in pairs_f]
                hyps = [tr[d][p[hyp_side]] for p in pairs_f]
                q[d] = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        f = {}
        for mode in ("fwd", "rev", "both"):
            c = counts.get((name, m, "chrf", mode))
            if c is not None:
                t = c.sum(0)
                f[mode] = 2 * t[2] / (t[0] + t[1])
        print(f"| {m} | {q.get('fwd', float('nan')):.1f} | " + (f"{q['rev']:.1f}" if "rev" in q else "-") + " | "
              + " | ".join(f"{f[k]:.3f}" if k in f else "-" for k in ("fwd", "rev", "both")) + " |")


def section3(name):
    counts = load_counts([f"{WORK}/all.json"])
    have = [m for m in MODELS if (name, m, "chrf", "both") in counts]
    n = len(counts[(name, have[0], "chrf", "both")])
    idx = resamples(n, 10000)
    print(f"\n#### {name}: paired bootstrap, chrF++ both directions, F1(row) - F1(column), 95 % interval\n")
    print("| | " + " | ".join(have) + " |")
    print("|---|" + "---|" * len(have))
    for a in have:
        cells = []
        for b in have:
            if a == b:
                cells.append("")
                continue
            d = f1s(counts[(name, a, "chrf", "both")], idx) - f1s(counts[(name, b, "chrf", "both")], idx)
            lo, hi = np.percentile(d, [2.5, 97.5])
            mark = " *" if lo > 0 or hi < 0 else ""
            cells.append(f"{np.mean(d):+.3f} ({lo:+.3f}, {hi:+.3f}){mark}")
        print(f"| {a} | " + " | ".join(cells) + " |")


def section4():
    print("\n#### Sentences left empty after a failed check, and the F1 when they are filled from another model\n")
    print("| set | model | direction | empty | strict F1 (chrF++) | with them filled by opus-mt |")
    print("|---|---|---|---|---|---|")
    for name in ("de-fr-dev", "de-fr-test", "en-is"):
        s = load(name)
        for m in MODELS:
            for d in ("fwd", "rev"):
                meta = f"{WORK}/{name}/{m}.{d}.json"
                if not os.path.exists(meta) or not json.load(open(meta))["failed"]:
                    continue
                tr = translations(name, m)
                own = (s.src if d == "fwd" else s.trg)
                bad = [i for i, (x, y) in enumerate(zip(own, tr[d])) if x.strip() and x.strip() != ".EOA" and not y.strip()]
                filled = dict(tr)
                filled[d] = [translations(name, "opus-mt")[d][i] if i in set(bad) else y for i, y in enumerate(tr[d])]
                f = []
                for t in (tr, filled):
                    r = runs(s, {d: t[d]})[d]
                    gs = [{(a, b) for a, b in g if a and b} for g in s.gold]
                    st = sum(len(g & {(a, b) for a, b, _ in x}) for g, x in check_pairs(gs, r, name))
                    f.append(2 * st / (sum(len(g) for g in gs) + sum(len(x) for x in r)))
                print(f"| {name} | {m} | {'source' if d == 'fwd' else 'target'} | {len(bad)} | {f[0]:.3f} | {f[1]:.3f} |")


if __name__ == "__main__":
    for name in ("de-fr-test", "en-is"):
        s = load(name)
        section1(name, s)
        section1b(name, s)
    for name in ("de-fr-dev", "de-fr-test", "en-is"):
        section2(name, load(name))
    for name in ("de-fr-test", "en-is"):
        section3(name)
    section4()
