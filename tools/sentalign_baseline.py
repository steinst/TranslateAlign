"""Score stored SentAlign 2 output (.path files, one per document) with translatealign.evaluate, so that both
aligners are scored by one function against one gold. Writes the per-document counts in the format of
evaluate_all.py under the model name `sentalign2` / `sentalign2-paper2023` (config `labse`, mode `fwd`).

    python tools/sentalign_baseline.py OUT.json --root /path/to/stored-output
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evalsets import check_pairs, load  # noqa: E402

from translatealign.evaluate import counts, read_alignment  # noqa: E402

SETS = {"de-fr-test": ("bleualign", [f"test{i}" for i in range(7)]), "en-is": ("parice", None)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--root", required=True, help="directory containing v2/ and paper/ baseline outputs")
    a = p.parse_args()
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for name, (folder, docs) in SETS.items():
        s = load(name)
        docs = docs or s.documents
        for label, version in (("sentalign2", "v2"), ("sentalign2-paper2023", "paper")):
            per_doc = [counts(g, read_alignment(f"{a.root}/{version}/{folder}/output/{d}.txt.path"))
                       for g, d in check_pairs(s.gold, docs, name, "documents named for SentAlign 2")]
            res[f"{name}|{label}|labse|fwd"] = per_doc
            t, g, st = (sum(c[k] for c in per_doc) for k in ("test", "gold", "strict"))
            print(name, label, "strict F1 %.3f" % (2 * st / (t + g)))
    json.dump(res, open(a.out, "w"))


if __name__ == "__main__":
    main()
