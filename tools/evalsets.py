"""The evaluation sets, loaded from the local copies (GPL-2.0 / CC BY 4.0, not copied into this repository).

    de-fr-dev   the Bleualign development document (1 document)          eval1957.{de,fr}, golddev.py
    de-fr-test  the Bleualign / Text+Berg test set (7 documents)         eval1989.{de,fr}, goldeval.py
    en-is       ParIce, 10 documents (SentAlign 2 evaluation data)       eval_data/parice/{eng,isl,gold}

Each set: text of the source side and of the target side with `.EOA` lines between documents, the gold per document.
Paths: TRANSLATEALIGN_BLEUALIGN_REPO (clone of Bleualign), TRANSLATEALIGN_SENTALIGN2 (default ../SentAlign2).
"""
import glob
import os
import re
import runpy
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(__file__))
from translatealign.evaluate import read_alignment  # noqa: E402

SENTALIGN2 = os.environ.get("TRANSLATEALIGN_SENTALIGN2", "../SentAlign2")
WORK = os.environ.get("TRANSLATEALIGN_EVAL7", "evaluation-output")


@dataclass
class EvalSet:
    name: str
    source_language: str
    target_language: str
    src: list      # lines, `.EOA` between documents
    trg: list
    gold: list     # per document: list of (source numbers, target numbers)
    documents: list  # names


def check_pairs(gold, docs, name, what="aligned documents"):
    """Gold and output are paired by position: refuse to pair them when their numbers differ (zip would drop some)."""
    if len(gold) != len(docs):
        raise ValueError(f"set {name}: {len(gold)} gold documents but {len(docs)} {what}")
    return list(zip(gold, docs))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def _bleualign(name, label, doc_names):
    repo = os.environ["TRANSLATEALIGN_BLEUALIGN_REPO"]
    ev = os.path.join(repo, "eval")
    if name == "eval1989":
        g = runpy.run_path(os.path.join(ev, "goldeval.py"))
        gold = [g["gold"][g["gold1990map"][d]] for d in range(len(g["gold1990map"]))]
    else:
        gold = [runpy.run_path(os.path.join(ev, "golddev.py"))["goldalign"]]
    gold = [[(tuple(s), tuple(t)) for s, t in d] for d in gold]
    return EvalSet(label, "de", "fr", _read(os.path.join(ev, name + ".de")), _read(os.path.join(ev, name + ".fr")),
                   gold, doc_names or [f"doc{i}" for i in range(len(gold))])


def _parice():
    base = os.path.join(SENTALIGN2, "eval_data", "parice")
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(base, "gold", "*.txt")))
    src, trg, gold = [], [], []
    for n, name in enumerate(names):
        for side, out in (("eng", src), ("isl", trg)):
            lines = _read(os.path.join(base, side, name + ".txt"))
            # the files carry a running number and a tab before every sentence
            out.extend(re.sub(r"^\d+\t", "", l) for l in lines)
            out.append(".EOA")
        gold.append(read_alignment(os.path.join(base, "gold", name + ".txt")))
    return EvalSet("en-is", "en", "is", src[:-1], trg[:-1], gold, names)


def load(name):
    if name == "de-fr-dev":
        return _bleualign("eval1957", name, None)
    if name == "de-fr-test":
        return _bleualign("eval1989", name, None)
    if name == "en-is":
        return _parice()
    raise ValueError(name)


SETS = ["de-fr-dev", "de-fr-test", "en-is"]
