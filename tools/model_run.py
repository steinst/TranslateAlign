"""Translate the German-French evaluation set with a model and align, end to end through the command line;
print strict and lax scores per scorer and direction, the translation speed and what the checks caught.

    TRANSLATEALIGN_BLEUALIGN_REPO=/path/to/Bleualign python tools/model_run.py \
        --model "Helsinki-NLP/opus-mt-{src}-{trg}" [--model-file FILE] [--set eval1957] [--document N]

--set eval1957 is the development document, eval1989 the test set (7 documents; --document N takes one of
them, counted from 1). The texts and the gold alignment are read from the clone (GPL-2.0, not copied here).
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from reproduce import EVAL, fmt, gold_documents  # noqa: E402

from translatealign import cli  # noqa: E402
from translatealign.evaluate import read_alignment, score  # noqa: E402
from translatealign.search import split_documents  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--model-file")
    p.add_argument("--set", default="eval1957", choices=["eval1957", "eval1989"])
    p.add_argument("--document", type=int, help="one document of the set, counted from 1")
    p.add_argument("--cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--keep", metavar="DIR", help="keep the output files here")
    a = p.parse_args()
    model = ["--model", a.model] if a.model else ["--model-file", a.model_file]
    cache = ["--no-cache"] if a.no_cache else (["--cache", a.cache] if a.cache else [])

    src = cli.read_lines(os.path.join(EVAL, a.set + ".de"))
    trg = cli.read_lines(os.path.join(EVAL, a.set + ".fr"))
    gold = gold_documents(a.set)
    if a.document:
        src = split_documents(src)[a.document - 1][1]
        trg = split_documents(trg)[a.document - 1][1]
        gold = [gold[a.document - 1]]
    tmp = a.keep or tempfile.mkdtemp()
    os.makedirs(tmp, exist_ok=True)
    fs, ft = os.path.join(tmp, "src.de"), os.path.join(tmp, "trg.fr")
    for path, lines in ((fs, src), (ft, trg)):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)
    n_docs = len(split_documents(src))
    print(f"{a.set}{' document %d' % a.document if a.document else ''}: {n_docs} document(s), "
          f"{len(src) - n_docs + 1} / {len(trg) - n_docs + 1} sentences; model {a.model or a.model_file}\n")
    print("| scorer | translated | beads | strict P / R / F1 | lax P / R / F1 | seconds |")
    print("|---|---|---|---|---|---|")
    for scorer in ("bleu", "chrf"):
        for which in ("source", "target", "both"):
            out = os.path.join(tmp, f"{scorer}.{which}")
            start = time.time()
            code = cli.main([fs, ft, "-o", out, "--scorer", scorer, "-s", "de", "-t", "fr", "--translate", which,
                             "--save-translations"] + model + cache)
            if code:
                sys.exit(code)
            docs = cli.per_document(read_alignment(out + ".path"), src, trg)
            strict, lax = fmt(score(gold, docs))
            print(f"| {scorer} | {which} | {sum(len(d) for d in docs)} | {strict} | {lax} | "
                  f"{time.time() - start:.1f} |", flush=True)


if __name__ == "__main__":
    main()
