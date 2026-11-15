# TranslateAlign – specification of the search

Written 14 Nov 2026 from the two publications only:

* **[SV10]** Rico Sennrich and Martin Volk (2010). MT-based Sentence Alignment for OCR-generated Parallel Texts.
  AMTA 2010. https://aclanthology.org/2010.amta-papers.14
* **[SV11]** Rico Sennrich and Martin Volk (2011). Iterative, MT-based Sentence Alignment of Parallel Texts.
  NODALIDA 2011. https://aclanthology.org/W11-4624

The Bleualign program (GPL-2.0) was **run** for reference output but its source code was **not read**; nothing in
TranslateAlign may be copied from it. Where the papers leave a detail open, this document says so, and the
reference output (section 6) is what TranslateAlign's choices are tested against.

Notation: S = source text, T = target text, T' = the machine translation of S into the target language, one line
per source sentence. Sentences are numbered from 0 within a document.

## 1. Input and units [SV10, section 4]
* One sentence per line. Documents are separated by **hard delimiters**; the search never crosses them and is run
  once per document. (The program uses a line `.EOA`.)
* Any number of translations can be given, in either direction (T' of S, or S' of T).
* Complexity is quadratic in the document length; the method was developed for documents of some hundred
  sentences.

## 2. Similarity score [SV10, 4.1]
BLEU between a sentence of T' and a sentence of T, with these changes:
1. n-grams up to **2** (not 4);
2. both sentences **tokenised and lowercased**;
3. computed **in both directions** (each sentence once as hypothesis, once as reference, because of the brevity
   penalty) and combined by the **harmonic mean**.
Unrelated sentences often score exactly 0; that is wanted.

TranslateAlign adds chrF++ as a second score, also in both directions with the harmonic mean (7.1); BLEU stays for
reproducing the papers. chrF++ is the default of the command line (`--scorer chrf`); `--scorer bleu` is the setting of the
papers.

## 3. First pass: anchor points [SV10, 4.2 and 4.3]
1. Score **all** pairs (t'_i, t_j): no pruning by position.
2. Keep the **3 best-scoring** candidates for each sentence.
3. The kept pairs are the vertices of a directed acyclic graph; an edge leads from v to w if both sentences of w
   come later in their texts than those of v. Find the path that **maximises the summed score** (edge weight = the
   negative score of the successor; shortest path over a topological order). Crossing links are therefore
   excluded.
4. Result: an ordered list of 1-to-1 beads, the **anchors**. The other sentences are unaligned for now.

## 4. Second pass: the gaps [SV10, 4.4]
For each pair of consecutive anchors (t'_i, t_j), (t'_k, t_l), the gap is the unaligned t'_x with i < x < k and t_y
with j < y < l; also before the first and after the last anchor. Per gap, in this order:

**4.1 One-to-n heuristic.**
* From t'_i, the gap's t'_x and t'_k make all sequences of 1, 2 or 3 sentences; the same for t_j, the gap's t_y and
  t_l. Score the Cartesian product of the two lists.
* If a bead ((t'_i, ..., t'_{i+n}), t_j) scores higher than (t'_i, t_j), it replaces (t'_i, t_j). If not, the
  analogous checks for t_j, t'_k and t_l.
* "Scores higher" means **both** a higher score **and** a higher absolute number of matches between the two
  sequences (against false positives from a smaller brevity penalty).
* If a bead was replaced, repeat this heuristic; else go on.

**4.2 One-to-one heuristic.** The bead (t'_{i+1}, t_{j+1}) is added if it is the best-scoring pair among all
(t'_{i+1}, t_y) of the gap (it may have been pruned in the first pass). If added, go back to 4.1; else go on.

**4.3 Gale and Church.** If the gap has size 0 on either side, or its two sides differ in size by more than a
factor of two ("except for two or three-word gaps"), what is left stays unaligned. Otherwise the length-based
algorithm of Gale and Church (1993) aligns everything left in the gap, on **T' and T** (the translation, not the
source).

[SV11, 4.1] summarises the second pass the same way: unaligned sentences are added to first-pass beads if the score
rises, aligned by a length-based algorithm if that keeps the order, or discarded.

## 5. Several runs [SV10, 4.5]
Run in both directions (T' against T, and S' against S) or with several translations, and **intersect**: keep only
the beads that are identical in every run. Higher precision, lower recall.

Not part of TranslateAlign: the iteration of [SV11] (training an SMT system on the aligned corpus and aligning
again) and its phrase-table pruning. A modern translation model takes the place of that system.

## 6. Reference output of the original program
Program: `rsennrich/Bleualign` at commit `5a11b7f`, run unchanged on the data it ships (`eval/`): the test set
`eval1989` (7 documents, 991 / 1,011 sentences, the Text+Berg set that Vecalign and SentAlign also use) and the
development document `eval1957` (468 / 554). Its verbose log gives, per document, the anchors of the first pass,
the final beads, and how many beads came from the first pass, the gap heuristics and Gale-Church.

| run | strict P / R / F1 | lax P / R / F1 | [SV10] table 4 (strict, lax F1) |
|---|---|---|---|
| test, translation `europarlfull` (de-fr) | 0.829 / 0.786 / 0.807 | 0.978 / 0.921 / 0.948 | 0.83 / 0.78 / 0.81, 0.95 |
| test, translation `google` (de-fr) | 0.831 / 0.788 / 0.809 | 0.978 / 0.923 / 0.950 | 0.83 / 0.78 / 0.81, 0.95 |
| test, translation `europarllight` (de-fr) | 0.722 / 0.608 / 0.660 | 0.906 / 0.769 / 0.832 | 0.72 / 0.60 / 0.66, 0.83 |
| test, both directions intersected (`europarlfull`) | 0.916 / 0.688 / 0.786 | 0.986 / 0.734 / 0.842 | 0.92 / 0.69 / 0.78, 0.84 |
| dev, translation `europarlfull` | 0.749 / 0.735 / 0.742 | 0.987 / 0.929 / 0.957 | (dev document, not in the table) |

The program reproduces the published table. Beads of the `europarlfull` test run: 813 (500 from the first pass
unchanged, 139 from the gap heuristics, 174 from Gale-Church); first-pass anchors: 627.

The reference files (bead numbers only, no text) are produced by `tools/extract_reference.py` and kept in
`tests/reference/`, with their provenance and licence described in `tests/reference/README.md`. The
evaluation texts and translations are part of the Bleualign repository (GPL-2.0) and are **not** copied here;
tests read them from a local clone selected with `TRANSLATEALIGN_BLEUALIGN_REPO`.

Scoring convention of the original (from its output, not its code): recall is counted over the gold beads that
have sentences on both sides; a hypothesis is a strict hit if identical to a gold bead, a lax hit if it overlaps
one on both sides.

## 7. What the papers leave open, and what was decided (14 Nov 2026)
Each point was settled by which choice reproduces the reference output (section 6), and where that did not
decide it, by running the original program on small made-up inputs and looking at its output ("probe"). Its
source code was not read. Code: `src/translatealign/search/`; numbers: section 7.9.

1. **BLEU details** (`BleuScorer.reference()`):
   * Tokeniser: the rules of the NIST script mteval-v13a, then lowercase (`scoring.bleu.tokenize_13a`; note that
     `_` is split off). Evidence: the original's log names the best candidate of every unaligned sentence; with
     this tokeniser all of them agree (364/364 test, 163/163 dev); whitespace splitting gives 352/364, the
     original tokenizer 290/364. `tokenize` stays the default of `BleuScorer()`.
   * No smoothing; a pair in which one side has a single token scores 0 (`effective_order=False`: probe and log:
     `wv` against `Wv` is not a candidate).
   * Both directions, harmonic mean. "Number of matches": clipped matches of all orders (`match_count="all"`);
     the choice against unigram matches was not needed to reach the agreement below and was not tested apart.
   * chrF++: without an original-program reference to compare with, but **in both directions with the harmonic mean**
     (`ChrfScorer(both_directions=True)`, what `--scorer chrf` uses): chosen on the development
     document; chrF++ in one direction favours recall and joins too many sentences (docs/RESULTS.md section 2).
2. **Pruning:** the 3 best target sentences **for each translated sentence** only (not for each target
   sentence), score > 0, ties to the lower sentence number. With both sides 623/627 anchors, with rows only 627/627.
3. **Path search:** maximum summed score; among equal sums the path found first (candidates in order of
   source then target number, a candidate takes the earliest of its equally good predecessors, the path ends
   at the earliest best candidate). The other tie rule loses 1 anchor in every test run.
4. **One-to-n heuristic** (all from probes; this is where the program differs most from the paper's text):
   * Gaps are visited once, in text order. The lists of a gap are fixed when it is visited: the bead before, the
     unaligned sentences, the bead after, on each side; a side of a bead that already has two sentences is one
     item. A **sequence is 1 or 2 neighbouring items** (at most 3 sentences), not "1, 2 or 3 sentences": a bead
     grows by one neighbour per gap, so a 3-sentence side arises from the gap before and the gap after.
   * Per bead (the one before the gap first, then the one after), the candidates are the pairs of sequences in
     which the source sequence is the bead's source side, or the target sequence is its target side, or both
     contain the bead's sentence. **Only the best-scoring candidate is looked at.** If it changes both sides
     (many-to-many) or takes a sentence of another bead, nothing happens to this bead: there is no fall-back
     to the next candidate. (Probe: a 2-to-1 join is made or not exactly according to whether a 2-to-2 pair
     with the next anchor's sentence scores higher; the bead on the other side of the gap is still handled.)
   * A sequence need not contain the bead's own sentence: the target sentence of an anchor can be replaced by
     two other sentences of the gap.
   * "Scores higher": a changed **source** side needs a higher score and more matches; a changed **target** side
     needs a higher score only (probes: joining unrelated text, which only improves the brevity penalty, is
     accepted on the target side and refused on the source side; fewer matches are accepted on the target side).
     The paper asks for both on both sides. Kept as found (`GapOptions.target_needs_more_matches=False`).
   * After a change the step is repeated with the same lists.
   * The gap after the last anchor is taken from the position of that anchor as it was after the first pass,
     even if the bead has moved (probe); the other gaps from the beads as they are when the gap is visited.
   * 2-to-2 beads in the output come from Gale-Church.
5. **One-to-one heuristic:** only the first unaligned pair of the gap (probe: not the last, not one in the
   middle); it is added if its score is above 0 and no pair of that translated sentence with a target
   *sequence* of the gap scores higher (probe: a long sentence is not aligned with a short one if the short one
   joined with its neighbour scores higher). No minimum score. The new bead stays inside the gap: it is not an
   anchor for the one-to-n step. Then the one-to-n step is repeated.
6. **Gale-Church condition:** both sides non-empty, and the larger side less than twice the smaller one, or at
   most 3 sentences (probed for all sizes 0..8 x 0..8: `search.gaps.size_rule`). Lengths in characters, the
   parameters of Gale and Church (1993).
   **The Gale-Church of the original is not the standard algorithm:** in its `--galechurch` mode the first
   sentence of a text never joins a 2-sentence bead (a text of 1 against 2 sentences gives `0 - 1`, not
   `0 - 0,1`), and other choices differ too. TranslateAlign uses the standard algorithm (the `galechurch`
   package, a required dependency) and does not imitate this. To test the heuristics apart from it, the output of the original's
   Gale-Church mode for each gap is recorded in `tests/reference/galechurch_gaps.json`.
7. **Intersection:** beads with the same source and the same target sentences in every run (`search.intersect`);
   a run in the other direction is turned round first (`flip`).
8. **Output:** `align` returns the beads with sentences on both sides; unaligned sentences are those in no bead.
   The command line writes `OUT.path` (`[0, 1]:[2]` lines, line numbers of the input files; delimiter
   lines are in no bead), `OUT.src`, `OUT.trg`; with `--unaligned` each unaligned sentence is a bead with one
   empty side, placed before the next bead (source sentences first).
9. **Scoring** (`translatealign.evaluate`): the convention at the end of section 6; it gives the original's own
   numbers on the original's output. Lax recall differs from the SentAlign / Vecalign script (docs/RESULTS.md).

### 7.9 Agreement with the original (`tools/agreement.py`, BLEU, 14 Nov 2026)
Identical beads / beads of the original; "ours" = number of beads TranslateAlign returns.

| run | anchors | beads, Gale-Church of the original for the gaps | beads, standard Gale-Church |
|---|---|---|---|
| test, `europarlfull` | 627 / 627 | 804 / 813 (ours 815) | 770 / 813 (ours 822) |
| test, `google` | 647 / 647 | 795 / 813 (ours 815) | 761 / 813 (ours 821) |
| test, `europarllight` | 406 / 406 | 700 / 723 (ours 731) | 651 / 723 (ours 745) |
| dev, `europarlfull` | 305 / 305 | 361 / 374 (ours 376) | 351 / 374 (ours 376) |
| test, both directions intersected | | 637 / 644 (ours 641) | 604 / 644 (ours 632) |

Beads per type, test `europarlfull`, ours / original: first pass unchanged 505 / 500, heuristics 134 / 139,
Gale-Church 176 / 174 (with the original's Gale-Church for the gaps).

Remaining differences with the original's Gale-Church (1 to 3 % of the beads): **unexplained.** The original
changes a few more anchors than this model (5 in the test run). Examples, test `europarlfull`: document 1
`207 - 182,183` (here the best candidate is a 2-to-2 pair made of gap sentences only, and the original takes the
1-to-2 pair below it, against the rule above, which a probe confirmed for a 2-to-2 pair that contains the next
anchor's sentence); document 3 `2,3 - 2,3`; document 5 `23,24,25 - 28,29,30` (a 3-to-3 bead, which neither the
heuristics as modelled nor Gale-Church can make). Variants tried and not better overall: checking the four
sides one after the other as the paper words it (804, 363, 795, 702 with a different set of errors), letting
only some many-to-many candidates block. Differences with the standard Gale-Church are the sum of these and of
the original's different Gale-Church; what they mean for precision and recall: docs/RESULTS.md section 1
(strict F1 within 0.008 of the original on the test set with good translations, 0.013 on the development
document, higher than the original with the weak translation).

### 7.10 The output is kept monotone (15 Nov 2026): a difference from the original
**Rule:** the one-to-n heuristic (point 4) accepts its best candidate only if the changed bead still lies
strictly between its neighbours on both sides: the bead or beads before it (the anchor before the gap, or the
last one-to-one bead added in the gap) and the one after it (the next one-to-one bead, the anchor after the gap,
or the end of the text). Otherwise nothing happens to this bead, as for a candidate that takes a sentence of
another bead (still only the best candidate is looked at; the tie rules are unchanged). So every result is
monotone: for consecutive beads a, b, `a.src[-1] < b.src[0]` and `a.trg[-1] < b.trg[0]`, and no sentence is in
two beads.

**Why it was needed:** the candidate lists of a gap are fixed when it is visited while the beads move. A bead can
leave a sentence, and the bead on the other side of the gap can then take it although it lies on the wrong side
of the first bead's new position. Example (BLEU, `tools/crossing_probe.py`, seed 0): translation
`d f | d | b a | c | e d d c e`, target `c d d f b | b a c e e`, anchors 0-0 and 4-1. Bead 0-0's best candidate is
4-0 (it takes the other anchor's sentence: refused); bead 4-1 then takes 2,3-1 (0.53 against 0.39, more
matches); now sentence 4 is free and bead 0-0 takes 4-0, which crosses 2,3-1. With the rule: 0-0 and 2,3-1.
(The code review of 15 Nov 2026 found the same with chrF++: anchors 0-0, 1-1, 2-2, 3-4 gave ... 2-3, 3-2.)

**What the original does (probed, its code not read):** it crosses too. On 100 random small problems on which
this implementation crossed before the fix (BLEU, vocabularies of 6 letters, at most 6 sentences per side; found
in 358,487 tries), the original's output was not monotone in 29: 31 crossings and in 3 problems a target
sentence in two beads; in all 29 it was identical to ours before the fix. Example: translation
`a d a f b a a f f | c c c e d`, target `e d a d | f c a d | c f b | f e b f | a d e d d`: the original and the old
code give `0 - 1,2` and `1 - 0`. On the example above the original gives `0,1 - 0` and `4 - 1` (monotone, but
not the same as the old code either; small problems with 6-letter vocabularies are where the remaining
differences of section 7.9 show most). Scripts, cases and the original's output:
`.../translatealign_work/job11/` (`run_original_batch.py`, `batch/bleu/bleu.json`).

**Decision:** keep the output monotone, against the original. A crossing alignment cannot be written as a
sequence of beads in text order (sorting the beads does not repair the links), the command-line output and
`--unaligned` assume the order, and a bead that crosses is a wrong link in any case. The one-to-one heuristic
and the length aligner work in `region()`, between the last bead before and the bead after, so they cannot
cross once the one-to-n step does not; tested, not assumed (below).

**Effect:** none on real text. Agreement with the original (section 7.9, `tools/agreement.py`), the reproduction
table (`tools/reproduce.py`) and all 92 entries of the model evaluation (`tools/evaluate_all.py` on the
stored translations, per-document counts) are identical before and after the change. Tests: the two cases above,
and a seeded property test (2,500 small problems for each of `BleuScorer.reference()` and
`ChrfScorer(both_directions=True)`, with empty and repeated sentences and several documents) that fails without
the rule; `tools/crossing_probe.py` found no failing problem in 400,000 more (both scorers, up to 8 sentences).

## 8. Consequences for the design
* The score is a function of (a sequence of translated sentences, a sequence of target sentences): scorers must
  accept **joined sentences**, and return the number of matches as well as the score (4.1).
* All pairs are scored (quadratic): the scorer must be vectorised; chrF++ and BLEU both reduce to n-gram counting,
  so one sparse matrix product per n-gram order gives all pairs.
* Three parts with clear borders: scorer, first pass (candidates + path), second pass (gap heuristics +
  Gale-Church). Gale-Church comes from the `galechurch` package (Apache-2.0).
* A test per stage against the reference: same anchors after the first pass; same beads at the end; same scores as
  the table in section 6.
