"""Speed and GPU memory of one model on one test document; one JSON line to stdout (run each model in its own
process so that the memory numbers are clean).

    python tools/model_speed.py --model nllb-200 --test de-fr
    python tools/model_speed.py --model-file my.toml --test en-is

--test de-fr: German side of test document 1 of the Bleualign set (needs TRANSLATEALIGN_BLEUALIGN_REPO or the
clone given in tools/reproduce.py), translated into French; --test fr-de: its French side into German;
--test en-is / is-en: ParIce document s_1 of the SentAlign 2 evaluation data (--parice DIR to change).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from reproduce import EVAL  # noqa: E402

from translatealign import cli  # noqa: E402
from translatealign.search import split_documents  # noqa: E402
from translatealign.translate import load_translator  # noqa: E402

PARICE = os.path.join(os.environ.get("TRANSLATEALIGN_SENTALIGN2", "../SentAlign2"), "eval_data", "parice")


def test_set(name, parice):
    if name in ("de-fr", "fr-de"):
        side = name.split("-")[0]
        lines = cli.read_lines(os.path.join(EVAL, "eval1989." + side))
        return split_documents(lines)[0][1], *name.split("-")
    src, trg = name.split("-")
    folder = {"en": "eng", "is": "isl"}[src]
    with open(os.path.join(parice, folder, "s_1.txt"), encoding="utf-8") as f:
        lines = [line.rstrip("\n").split("\t", 1)[-1] for line in f]
    return lines, src, trg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--model-file")
    p.add_argument("--test", required=True, choices=["de-fr", "fr-de", "en-is", "is-en"])
    p.add_argument("--parice", default=PARICE)
    a = p.parse_args()
    sentences, src, trg = test_set(a.test, a.parice)
    sentences = [s for s in sentences if s.strip()]
    import torch
    t = load_translator(a.model, a.model_file)
    result = {"model": a.model or a.model_file, "test": a.test, "sentences": len(sentences),
              "characters": sum(map(len, sentences))}
    try:
        start = time.time()
        t.translate(sentences[:1], src, trg)          # loads the model
        result["load_and_first_s"] = round(time.time() - start, 1)
        if torch.cuda.is_available():
            result["gpu_after_load_gb"] = round(torch.cuda.memory_allocated() / 2**30, 1)
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        out = t.translate(sentences, src, trg)
        seconds = time.time() - start
        result.update(seconds=round(seconds, 1), sentences_per_s=round(len(sentences) / seconds, 2),
                      book_7000_min=round(7000 / (len(sentences) / seconds) / 60, 1),
                      problems=len(out.problems), retried=out.retried if isinstance(out.retried, int) else len(out.retried),
                      empty=sum(1 for x in out if not x.strip()))
        if torch.cuda.is_available():
            result["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        result["examples"] = [[s, o] for s, o in list(zip(sentences, out))[:3]]
        result["problem_reasons"] = [str(x) for x in out.problems[:3]]
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
