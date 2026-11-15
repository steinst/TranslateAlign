"""Loading the evaluation data of the original Bleualign (from a local clone, never copied here) and the
reference output in tests/reference/."""
import json
import os

HERE = os.path.dirname(__file__)

# run name -> (translated side, its translation, other side, reference file, flipped)
# flipped: the French side was translated to German and is aligned with the German text.
RUNS = {
    "test_europarlfull": ("eval1989.de", "eval1989.europarlfull.fr", "eval1989.fr", "test_europarlfull", False),
    "test_google": ("eval1989.de", "eval1989.google.fr", "eval1989.fr", "test_google", False),
    "test_europarllight": ("eval1989.de", "eval1989.europarllight.fr", "eval1989.fr", "test_europarllight", False),
    "dev_europarlfull": ("eval1957.de", "eval1957.europarlfull.fr", "eval1957.fr", "dev_europarlfull", False),
    "test_europarlfull_reverse": ("eval1989.fr", "eval1989.europarlfull.de", "eval1989.de", None, True),
}


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def _split(lines, marks):
    docs = [[]]
    for line, mark in zip(lines, marks):
        if mark.strip() == ".EOA":
            docs.append([])
        else:
            docs[-1].append(line)
    return docs


def load_run(run, repo):
    """Per document: (translated sentences, sentences of the other side, reference entry or None)."""
    side, translation, other, ref_name, _ = RUNS[run]
    ev = os.path.join(repo, "eval")
    s, x, t = _lines(os.path.join(ev, side)), _lines(os.path.join(ev, translation)), _lines(os.path.join(ev, other))
    assert len(s) == len(x)
    ref = reference(ref_name) if ref_name else None
    xd, td = _split(x, s), _split(t, t)
    return [(xd[d], td[d], ref[d] if ref else None) for d in range(len(xd))]


def reference(name):
    with open(os.path.join(HERE, "reference", name + ".json")) as f:
        return json.load(f)
