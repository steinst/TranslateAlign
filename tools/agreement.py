"""Agreement of TranslateAlign's search with the reference output of the original Bleualign, per run:
anchors, final beads, counts per type; with the original's Gale-Church (recorded, see record_gc_oracle.py)
and with the standard one.

    TRANSLATEALIGN_BLEUALIGN_REPO=/path/to/Bleualign python tools/agreement.py
"""
import hashlib, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from evalsets import check_pairs  # noqa: E402
from reference_data import HERE, RUNS, load_run, reference  # noqa: E402

from translatealign.scoring import BleuScorer  # noqa: E402
from translatealign.search import align, find_anchors, flip, intersect, galechurch_aligner  # noqa: E402

REPO = os.environ["TRANSLATEALIGN_BLEUALIGN_REPO"]
STORE = json.load(open(os.path.join(HERE, "reference", "galechurch_gaps.json")))
KINDS = {"anchor": "BLEU", "heuristic": "GAPFILLER", "length": "GALECHURCH"}


def recorded_aligner(src, trg):
    k = hashlib.sha1(("\n".join(src) + "\n\t\n" + "\n".join(trg)).encode("utf-8")).hexdigest()
    return [tuple(b) for b in STORE[k]]


def run_stats(run, aligner):
    scorer = BleuScorer.reference()
    st = dict(anchors_ref=0, anchors_same=0, anchors_ours=0, beads_ref=0, beads_ours=0, beads_same=0,
              docs=0, docs_same=0, types={}, types_ref={})
    out = []
    for translated, target, ref in load_run(run, REPO):
        beads = align(translated, target, scorer, length_aligner=aligner)
        out.append(beads)
        if ref is None:
            continue
        a = set(find_anchors(scorer, translated, target))
        ra = {tuple(p) for p in ref["anchors"]}
        ours = {(b.src, b.trg) for b in beads}
        rb = {(tuple(s), tuple(t)) for s, t in ref["beads"]}
        st["anchors_ref"] += len(ra); st["anchors_ours"] += len(a); st["anchors_same"] += len(a & ra)
        st["beads_ref"] += len(rb); st["beads_ours"] += len(ours); st["beads_same"] += len(ours & rb)
        st["docs"] += 1; st["docs_same"] += ours == rb
        for b in beads:
            st["types"][KINDS[b.kind]] = st["types"].get(KINDS[b.kind], 0) + 1
        for k, v in ref["types"].items():
            st["types_ref"][k] = st["types_ref"].get(k, 0) + v
    return st, out


def both_directions(aligner):
    _, fwd = run_stats("test_europarlfull", aligner)
    _, rev = run_stats("test_europarlfull_reverse", aligner)
    ref = reference("test_europarlfull_both_directions")
    same = n_ref = n_ours = 0
    check_pairs(ref, fwd, "reference output", "aligned documents")
    check_pairs(fwd, rev, "forward and reverse runs", "documents")
    for d, (f, r) in enumerate(zip(fwd, rev)):
        ours = {(b.src, b.trg) for b in intersect(f, flip(r))}
        rb = {(tuple(s), tuple(t)) for s, t in ref[d]["beads"]}
        same += len(ours & rb); n_ref += len(rb); n_ours += len(ours)
    return n_ref, n_ours, same


if __name__ == "__main__":
    for label, aligner in (("Gale-Church of the original (recorded)", recorded_aligner),
                           ("standard Gale-Church (the galechurch package, the default)", galechurch_aligner)):
        print("==", label)
        for run in RUNS:
            if RUNS[run][3] is None:
                continue
            st, _ = run_stats(run, aligner)
            print("  %-20s anchors %d/%d (ours %d) | beads %d/%d (ours %d), documents identical %d/%d | types ours %s ref %s" % (
                run, st["anchors_same"], st["anchors_ref"], st["anchors_ours"], st["beads_same"], st["beads_ref"],
                st["beads_ours"], st["docs_same"], st["docs"], dict(sorted(st["types"].items())), dict(sorted(st["types_ref"].items()))))
        try:
            print("  both directions intersected: reference %d, ours %d, identical %d" % both_directions(aligner))
        except KeyError:
            print("  both directions intersected: gaps of the reverse run are not recorded")
