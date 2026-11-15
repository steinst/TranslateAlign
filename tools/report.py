"""Tables from the per-document counts of evaluate_all.py / sentalign_baseline.py (markdown on stdout).

    python tools/report.py RESULTS.json [RESULTS2.json ...] [--set de-fr-test] [--boot 10000]

Strict F1 is computed from counts summed over documents; the interval is a percentile bootstrap over documents
(documents resampled with replacement; 7 documents in the German-French test set, 10 in ParIce, so the intervals
are wide). Differences between two configurations are bootstrapped on the same resamples (paired).
"""
import argparse
import json
import sys

import numpy as np

FIELDS = ["test", "gold", "strict", "lax_test", "lax_gold"]


def load(paths):
    res = {}
    for p in paths:
        res.update(json.load(open(p)))
    return {tuple(k.split("|")): np.array([[d[f] for f in FIELDS] for d in v]) for k, v in res.items()}


def prf(c):
    """c: summed counts -> (P, R, F1) strict, F1 lax."""
    t, g, s, lt, lg = c
    p, r = (s / t if t else 0.0), (s / g if g else 0.0)
    f = 2 * p * r / (p + r) if p + r else 0.0
    lp, lr = (lt / t if t else 0.0), (lg / g if g else 0.0)
    lf = 2 * lp * lr / (lp + lr) if lp + lr else 0.0
    return p, r, f, lf


def f1s(counts, idx):
    """Strict F1 for each resample: idx (B, n) of document numbers."""
    c = counts[idx].sum(axis=1)
    t, g, s = c[:, 0], c[:, 1], c[:, 2]
    return np.where(t + g > 0, 2 * s / np.maximum(t + g, 1), 0.0)      # F1 = 2 s / (t + g)


def resamples(n, boot, seed=1):
    return np.random.default_rng(seed).integers(0, n, size=(boot, n))


def row(counts, idx):
    p, r, f, lf = prf(counts.sum(axis=0))
    lo, hi = np.percentile(f1s(counts, idx), [2.5, 97.5])
    return p, r, f, lf, lo, hi


def main():
    a = argparse.ArgumentParser()
    a.add_argument("results", nargs="+")
    a.add_argument("--set", default=None)
    a.add_argument("--boot", type=int, default=10000)
    a.add_argument("--vs", default="sentalign2|labse|fwd", help="model|config|mode to compare against")
    a.add_argument("--configs", nargs="*", default=None)
    args = a.parse_args()
    data = load(args.results)
    sets = [args.set] if args.set else sorted({k[0] for k in data})
    for s in sets:
        keys = [k for k in data if k[0] == s and (not args.configs or k[2] in args.configs or k[1].startswith("sentalign"))]
        n = len(next(iter(data[k] for k in keys)))
        idx = resamples(n, args.boot)
        ref = None
        if args.vs:
            m, c, mode = args.vs.split("|")
            ref = data.get((s, m, c, mode))
        print(f"\n### {s} ({n} document{'s' if n > 1 else ''}; strict, bootstrap over documents)\n")
        print("| model | scorer | translated | beads | P | R | F1 | 95 % interval | lax F1 | F1 - SentAlign 2 (interval) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k in sorted(keys, key=lambda k: (k[1].startswith("sentalign"), k[1], k[2], ["fwd", "rev", "both"].index(k[3]))):
            cnt = data[k]
            p, r, f, lf, lo, hi = row(cnt, idx)
            diff = ""
            if ref is not None and k[1] != "sentalign2":
                d = f1s(cnt, idx) - f1s(ref, idx)
                diff = "%+.3f (%+.3f, %+.3f)" % (prf(cnt.sum(0))[2] - prf(ref.sum(0))[2], *np.percentile(d, [2.5, 97.5]))
            what = {"fwd": "source", "rev": "target", "both": "both, intersected"}[k[3]] if not k[1].startswith("sentalign") else "-"
            print(f"| {k[1]} | {k[2]} | {what} | {int(cnt[:, 0].sum())} | {p:.3f} | {r:.3f} | **{f:.3f}** | "
                  f"{lo:.3f} - {hi:.3f} | {lf:.3f} | {diff} |")


if __name__ == "__main__":
    main()
