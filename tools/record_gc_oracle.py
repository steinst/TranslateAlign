"""Record what the Gale-Church mode of the original Bleualign returns for the gaps that TranslateAlign's
second pass hands to its length aligner on the evaluation data, so that the reference tests can check the
heuristics separately from the length aligner. The original is run as a black box
(`bleualign.py --galechurch --srctotarget -`); its code is not read. Output: sentence numbers only, keyed by a
hash of the gap's text: tests/reference/galechurch_gaps.json.

    TRANSLATEALIGN_BLEUALIGN_REPO=/path/to/Bleualign python tools/record_gc_oracle.py
"""
import hashlib, json, os, re, shutil, subprocess, sys, tempfile

REPO = os.environ["TRANSLATEALIGN_BLEUALIGN_REPO"]
OUT = os.path.join(os.path.dirname(__file__), "..", "tests", "reference", "galechurch_gaps.json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from reference_data import RUNS, load_run  # noqa: E402

from translatealign.scoring import BleuScorer  # noqa: E402
from translatealign.search import align  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def key(src, trg):
    return hashlib.sha1(("\n".join(src) + "\n\t\n" + "\n".join(trg)).encode("utf-8")).hexdigest()


def original_galechurch(src, trg):
    d = tempfile.mkdtemp()
    try:
        for name, lines in (("s", src), ("t", trg)):
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write("".join(line + "\n" for line in lines))
        r = subprocess.run([sys.executable, "bleualign.py", "-s", d + "/s", "-t", d + "/t", "-o", d + "/o",
                            "--srctotarget", "-", "--galechurch", "-v", "2", "-p", "1"],
                           cwd=REPO, capture_output=True, text=True)
        beads = []
        for line in ANSI.sub("", r.stdout).split("\n"):
            if line.startswith("alignment: "):
                s, t = line[len("alignment: "):].split(" - ")
                beads.append([[int(v) for v in s.split(",") if v], [int(v) for v in t.split(",") if v]])
        return beads
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    try:
        store = json.load(open(OUT))
    except FileNotFoundError:
        store = {}
    new = 0

    def aligner(src, trg):
        nonlocal new
        k = key(src, trg)
        if k not in store:
            store[k] = original_galechurch(list(src), list(trg))
            new += 1
        return [tuple(b) for b in store[k]]

    scorer = BleuScorer.reference()
    for run in RUNS:
        for translated, target, _ in load_run(run, REPO):
            align(translated, target, scorer, length_aligner=aligner)
    json.dump(store, open(OUT, "w"), sort_keys=True)
    print(len(store), "gaps stored,", new, "new")


if __name__ == "__main__":
    main()
