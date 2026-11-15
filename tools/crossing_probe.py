"""Search random small problems for a result that is not monotone (docs/SPEC.md section 7.10).

A result is monotone if for consecutive beads a, b: a.src[-1] < b.src[0] and a.trg[-1] < b.trg[0], and no
sentence is in two beads. Prints the first failing problem per scorer, minimised (sentences and words removed
while it still fails), and writes it as files for the original program:

    python tools/crossing_probe.py [--scorer bleu|chrf] [--tries N] [--seed S] [--write DIR]

With --write DIR: DIR/x.src (the "source", here identical to the translation), DIR/x.trg, DIR/x.tr (the
translation of the source, line by line), to be run with
    python3 bleualign.py -s DIR/x.src -t DIR/x.trg --srctotarget DIR/x.tr -o DIR/out --printempty -v 2
"""
import argparse
import os
import random

from translatealign.scoring import BleuScorer, ChrfScorer
from translatealign.search import align, find_anchors


def problems(beads):
    out = []
    for side in ("src", "trg"):
        used = [i for b in beads for i in getattr(b, side)]
        if len(used) != len(set(used)):
            out.append(f"a {side} sentence in two beads")
    for a, b in zip(beads, beads[1:]):
        if not (a.src[-1] < b.src[0] and a.trg[-1] < b.trg[0]):
            out.append(f"crossing: {a} then {b}")
    return out


def make(rnd, vocab_size, max_sents, max_words):
    vocab = "abcdefghijklmnopqrstuvwxyz"[:vocab_size]

    def sent():
        return " ".join(rnd.choice(vocab) for _ in range(rnd.randint(1, max_words)))
    return [sent() for _ in range(rnd.randint(1, max_sents))], [sent() for _ in range(rnd.randint(1, max_sents))]


def fails(x, t, scorer):
    return bool(problems(align(x, t, scorer)))


def minimise(x, t, scorer):
    changed = True
    while changed:
        changed = False
        for side in (0, 1):
            cur = [x, t][side]
            for i in range(len(cur)):          # drop a sentence
                cand = cur[:i] + cur[i + 1:]
                if cand and fails(*((cand, t) if side == 0 else (x, cand)), scorer):
                    x, t = (cand, t) if side == 0 else (x, cand)
                    changed = True
                    break
            if changed:
                break
            for i, s in enumerate(cur):        # drop a word
                w = s.split()
                for k in range(len(w)):
                    if len(w) == 1:
                        break
                    cand = list(cur)
                    cand[i] = " ".join(w[:k] + w[k + 1:])
                    if fails(*((cand, t) if side == 0 else (x, cand)), scorer):
                        x, t = (cand, t) if side == 0 else (x, cand)
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
    return x, t


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scorer", choices=["bleu", "chrf"], default="bleu")
    p.add_argument("--tries", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vocab", type=int, default=6)
    p.add_argument("--max-sents", type=int, default=6)
    p.add_argument("--max-words", type=int, default=9)
    p.add_argument("--write")
    a = p.parse_args(argv)
    scorer = BleuScorer.reference() if a.scorer == "bleu" else ChrfScorer(both_directions=True)
    rnd = random.Random(a.seed)
    for it in range(a.tries):
        x, t = make(rnd, a.vocab, a.max_sents, a.max_words)
        if not fails(x, t, scorer):
            continue
        print(f"failing problem at try {it} (seed {a.seed})")
        x, t = minimise(x, t, scorer)
        print("translation:", x)
        print("target:     ", t)
        print("anchors:    ", find_anchors(scorer, x, t, 3))
        beads = align(x, t, scorer)
        print("beads:      ", [(b.src, b.trg, b.kind) for b in beads])
        print("\n".join(problems(beads)))
        if a.write:
            os.makedirs(a.write, exist_ok=True)
            for name, lines in (("x.src", x), ("x.tr", x), ("x.trg", t)):
                with open(os.path.join(a.write, name), "w", encoding="utf-8") as f:
                    f.write("".join(s + "\n" for s in lines))
        return 1
    print(f"no failing problem in {a.tries} tries (seed {a.seed}, scorer {a.scorer})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
