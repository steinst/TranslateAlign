"""The table of docs/RESULTS.md: the runs of the Bleualign paper (three translations, both directions
intersected, the development document), end to end through the command line, scored against the gold
alignment; next to the original program's output (tests/reference) and to TranslateAlign's search with the
original's Gale-Church for the gaps (recorded, see record_gc_oracle.py).

    TRANSLATEALIGN_BLEUALIGN_REPO=/path/to/Bleualign python tools/reproduce.py

The evaluation texts and the gold alignment are read from the clone (GPL-2.0, not copied here).
"""
import os
import runpy
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from agreement import recorded_aligner, run_stats  # noqa: E402
from reference_data import reference  # noqa: E402
from evalsets import check_pairs  # noqa: E402

from translatealign import cli  # noqa: E402
from translatealign.evaluate import read_alignment, score  # noqa: E402
from translatealign.search import flip, intersect  # noqa: E402

REPO = os.environ["TRANSLATEALIGN_BLEUALIGN_REPO"]
EVAL = os.path.join(REPO, "eval")

# label, set, translations of the source, translations of the target, reference file, runs for the recorded column
RUNS = [
    ("test, `europarlfull`", "eval1989", ["europarlfull"], [], "test_europarlfull", ["test_europarlfull"]),
    ("test, `google`", "eval1989", ["google"], [], "test_google", ["test_google"]),
    ("test, `europarllight`", "eval1989", ["europarllight"], [], "test_europarllight", ["test_europarllight"]),
    ("test, both directions intersected (`europarlfull`)", "eval1989", ["europarlfull"], ["europarlfull"],
     "test_europarlfull_both_directions", ["test_europarlfull", "test_europarlfull_reverse"]),
    ("dev, `europarlfull`", "eval1957", ["europarlfull"], [], "dev_europarlfull", ["dev_europarlfull"]),
]


def gold_documents(name):
    """The gold alignment shipped with the original program, per document in file order."""
    if name == "eval1989":
        g = runpy.run_path(os.path.join(EVAL, "goldeval.py"))
        return [g["gold"][g["gold1990map"][d]] for d in range(len(g["gold1990map"]))]
    return [runpy.run_path(os.path.join(EVAL, "golddev.py"))["goldalign"]]


def fmt(res):
    return "%.3f / %.3f / %.3f" % (res["precision_strict"], res["recall_strict"], res["f1_strict"]), \
           "%.3f / %.3f / %.3f" % (res["precision_lax"], res["recall_lax"], res["f1_lax"])


def same(ref, docs):
    pairs = check_pairs(ref, docs, "reference output", "aligned documents")
    return sum(len({(tuple(s), tuple(t)) for s, t in r["beads"]} & set(d)) for r, d in pairs)


def command_line(name, fwd, rev, scorer, tmp):
    src, trg = os.path.join(EVAL, name + ".de"), os.path.join(EVAL, name + ".fr")
    argv = [src, trg, "-o", os.path.join(tmp, "out"), "--scorer", scorer, "-q"]
    for x in fwd:
        argv += ["--translation", os.path.join(EVAL, f"{name}.{x}.fr")]
    for x in rev:
        argv += ["--translation-of-target", os.path.join(EVAL, f"{name}.{x}.de")]
    assert cli.main(argv) == 0
    beads = read_alignment(os.path.join(tmp, "out.path"))
    return cli.per_document(beads, cli.read_lines(src), cli.read_lines(trg))


def plain_chrf(name, fwd, rev):
    """chrF++ in one direction only (the translation as hypothesis), through the same function as the
    command line."""
    from translatealign.scoring import ChrfScorer
    src, trg = cli.read_lines(os.path.join(EVAL, name + ".de")), cli.read_lines(os.path.join(EVAL, name + ".fr"))
    beads = cli.align_files(src, trg, [cli.read_lines(os.path.join(EVAL, f"{name}.{x}.fr")) for x in fwd],
                            [cli.read_lines(os.path.join(EVAL, f"{name}.{x}.de")) for x in rev], scorer=ChrfScorer())
    return cli.per_document([(b.src, b.trg) for b in beads], src, trg)


def recorded(runs):
    """The library with the original's Gale-Church for the gaps (BLEU)."""
    outs = [run_stats(r, recorded_aligner)[1] for r in runs]
    if len(outs) == 2:
        outs = [[intersect(f, flip(r)) for f, r in zip(*outs)]]
    return [[(b.src, b.trg) for b in doc] for doc in outs[0]]


if __name__ == "__main__":
    rows = {"original": [], "recorded": [], "bleu": [], "chrf": [], "plain": []}
    with tempfile.TemporaryDirectory() as tmp:
        for label, name, fwd, rev, ref_name, rec_runs in RUNS:
            gold, ref = gold_documents(name), reference(ref_name)
            n_ref = sum(len(r["beads"]) for r in ref)
            rows["original"].append((label, score(gold, [r["beads"] for r in ref]), n_ref, n_ref))
            docs = recorded(rec_runs)
            rows["recorded"].append((label, score(gold, docs), same(ref, docs), n_ref))
            for scorer in ("bleu", "chrf"):
                docs = command_line(name, fwd, rev, scorer, tmp)
                rows[scorer].append((label, score(gold, docs), same(ref, docs), n_ref))
            docs = plain_chrf(name, fwd, rev)
            rows["plain"].append((label, score(gold, docs), same(ref, docs), n_ref))
    titles = {"original": "The original Bleualign (its output, scored here)",
              "recorded": "TranslateAlign, BLEU, Gale-Church of the original for the gaps (library, not the command line)",
              "bleu": "TranslateAlign, `--scorer bleu` (command line; standard Gale-Church)",
              "chrf": "TranslateAlign, `--scorer chrf` (command line; chrF++ in both directions; standard Gale-Church)",
              "plain": "TranslateAlign, chrF++ in one direction (library; not offered on the command line)"}
    for key in rows:
        print("\n**%s**\n" % titles[key])
        print("| run | beads | strict P / R / F1 | lax P / R / F1 | beads identical to the original's |")
        print("|---|---|---|---|---|")
        for label, res, n_same, n_ref in rows[key]:
            s, l = fmt(res)
            print("| %s | %d | %s | %s | %s |" % (label, res["test"], s, l,
                                                 "" if key == "original" else "%d / %d" % (n_same, n_ref)))
