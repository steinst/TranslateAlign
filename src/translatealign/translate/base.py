"""The translator interface: sentences in, the same number of sentences out.

A backend implements `translate_batch` (one call of the model). `Translator.translate` does the rest: lines that
need no translation, the cache, each distinct sentence once, batches of similar length, the checks on every
output, one retry with other settings, and the fallback to an empty translation (marked, with a warning) so that
the aligner can still run.
"""
from __future__ import annotations

import hashlib
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .cache import TranslationCache

# Commentary that a chat model puts round a translation. A pattern only counts if it does not match the
# source sentence too (a source that itself begins "Translation:" is translated to something like it).
DEFAULT_LEAK_PATTERNS = (
    r"^\W*(here is|here's|here are|below is)\b[^.:\n]{0,60}\btranslat",
    # "Of course I ..." / "Okay, I ..." / "I am unable to provide ..." are also ordinary sentences, so these two
    # patterns ask for words about the translation itself (earlier patterns rejected real translations).
    r"^\W*(sure|certainly|of course|okay|ok|absolutely)\b[,!.]?\s+(here('s| is| are)\b[^.\n]{0,60}"
    r"(translat|:)|here you go\s*:|the translation\b|let me translate|i('ll| will| can) translate)",
    r"^\W*(the\s+)?translat(ion|ed text|ed sentence)\b[^:\n]{0,40}:",
    r"^\W*(note|explanation)\s*:",
    r"\(\s*note\s*:",
    r"\bas an ai\b",
    r"\bi (cannot|can't|can not|am unable to|'m unable to) (translate|provide (a|the|any) translation|"
    r"(help|assist) (you )?with (this|the|that) translation)",
    r"<\|?(im_start|im_end|start_of_turn|end_of_turn|endoftext|eot_id)\|?>",
    r"<think>",
)

SEVERAL_LINES = "several lines"
EMPTY = "empty output"
LEAK = "prompt leaked"
COMMENTARY = "commentary"
TOO_LONG = "too long"
TOO_SHORT = "too short"
COUNT = "wrong number of outputs"


@dataclass
class Checks:
    """What every translated sentence must pass. Lengths are counted in characters:
    min_ratio * len(source) - slack <= len(output) <= max_ratio * len(source) + slack."""
    min_ratio: float = 1 / 6
    max_ratio: float = 6.0
    slack: int = 10
    leak_patterns: Sequence[str] = DEFAULT_LEAK_PATTERNS
    prompt_fragments: Sequence[str] = ()      # literal pieces of the prompt; none may show up in an output

    def __post_init__(self):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.leak_patterns]
        self._fragments = [f.lower() for f in self.prompt_fragments if f.strip()]

    def fingerprint(self) -> str:
        """A hash of all the settings above: a failure recorded in the cache counts only under the same one."""
        fields = [self.min_ratio, self.max_ratio, self.slack, list(self.leak_patterns), list(self.prompt_fragments)]
        return hashlib.sha256(json.dumps(fields).encode("utf-8")).hexdigest()[:16]

    def problem(self, source: str, output: str) -> Optional[str]:
        """None if the output is acceptable, else the reason."""
        out = output.strip()
        if not out:
            return EMPTY
        if "\n" in out or "\r" in out:
            return SEVERAL_LINES
        low_out, low_src = out.lower(), source.lower()
        if any(f in low_out and f not in low_src for f in self._fragments):
            return LEAK
        if any(p.search(out) and not p.search(source) for p in self._patterns):
            return COMMENTARY
        if len(out) > self.max_ratio * len(source) + self.slack:
            return TOO_LONG
        if len(out) < self.min_ratio * len(source) - self.slack:
            return TOO_SHORT
        return None


@dataclass
class Problem:
    """A sentence that got an empty translation because both attempts failed a check."""
    lines: tuple            # the line numbers (from 0) that hold this sentence
    source: str
    reason: str             # of the second attempt
    first_reason: str
    output: str             # what the model gave at the second attempt


class Translations(list):
    """The translated sentences (a list of str, one per input sentence) with a record of what happened."""

    def __init__(self, items=()):
        super().__init__(items)
        self.problems: list = []    # Problem: sentences left empty
        self.from_cache = 0         # distinct sentences found in the cache
        self.translated = 0         # distinct sentences sent to the model
        self.retried = 0            # of those, sentences that needed the second attempt
        self.known_failures = 0     # distinct sentences that the cache knows to fail (not tried again)
        self.skipped = 0            # lines not translated: empty lines and lines in `keep`
        self.rechecked = 0          # distinct cached translations that failed the current checks (translated again)
        self.truncated: list = []   # sentences whose translation stopped at max_new_tokens (may be cut off)


def make_batches(texts: Sequence[str], batch_size: int, max_batch_chars: int) -> list:
    """Texts sorted from long to short, cut into batches of at most `batch_size` sentences whose padded size
    (number of sentences x the longest one) stays within `max_batch_chars`; a batch has at least one."""
    ordered = sorted(texts, key=lambda t: (-len(t), t))
    batches, cur = [], []
    for t in ordered:
        if cur and (len(cur) >= batch_size or (len(cur) + 1) * max(len(cur[0]), 1) > max_batch_chars):
            batches.append(cur)
            cur = []
        cur.append(t)
    if cur:
        batches.append(cur)
    return batches


class Translator:
    """Base class. A backend sets `name`, implements `translate_batch` and, if its output depends on
    settings, `cache_id`."""

    name = "translator"
    batch_size = 32
    max_batch_chars = 12000
    checks = Checks()
    # A backend sets this in translate_batch: the positions, in the batch it just translated, of outputs
    # that stopped because the limit on new tokens was reached (the translation may be cut off).
    last_truncated: Sequence[int] = ()

    def cache_id(self, source_language: str, target_language: str) -> str:
        """Identifies everything that decides the output apart from languages and text (model, settings)."""
        return self.name

    def translate_batch(self, sentences: Sequence[str], source_language: str, target_language: str,
                        attempt: int = 0) -> list:
        """One call of the model: a list with one string per sentence. `attempt` is 0 for the normal
        settings and 1 for the retry settings (the batch then has one sentence)."""
        raise NotImplementedError

    def close(self) -> None:
        """Free the model."""

    def translate(self, sentences: Sequence[str], source_language: str, target_language: str, *,
                  cache: Optional[TranslationCache] = None, keep: Iterable[str] = (),
                  retry_failed: bool = False, progress: Optional[Callable[[int, int], None]] = None) -> Translations:
        """Translate line by line; the result has exactly as many items as `sentences`.

        cache:    translations are read from it and written to it after every batch, so an interrupted run
                  goes on where it stopped and nothing is translated twice.
        keep:     lines returned unchanged (document delimiters). Empty lines stay empty.
        retry_failed: try again the sentences that the cache remembers as failed (default: they get their
                  empty translation without the model being called). A failure counts only if it was
                  recorded under the same checks; a cached translation that fails the current checks is
                  translated again (`.rechecked`).
        progress: called with (distinct sentences done, distinct sentences to do) after every batch.
        A sentence that fails the checks twice becomes "" and is listed in `.problems` of the result.
        """
        keep = {k.strip() for k in keep}
        out = Translations([""] * len(sentences))
        where: dict = {}
        for i, s in enumerate(sentences):
            text = s.strip()
            if not text:
                out.skipped += 1
            elif text in keep:
                out[i] = s
                out.skipped += 1
            else:
                where.setdefault(text, []).append(i)

        def store(text, translation):
            for i in where[text]:
                out[i] = translation

        self._cut = set()
        ident = self.cache_id(source_language, target_language)
        fingerprint = self.checks.fingerprint()
        todo = list(where)
        if cache is not None:
            found = cache.get_many(ident, source_language, target_language, todo)
            # the current checks apply to a cached translation too (they are not part of the cache key)
            stale = [t for t, tr in found.items() if self.checks.problem(t, tr) is not None]
            for text in stale:
                del found[text]
            out.rechecked = len(stale)
            for text, translation in found.items():
                store(text, translation)
            out.from_cache = len(found)
            todo = [t for t in todo if t not in found]
            if not retry_failed:
                known = cache.get_failures(ident, source_language, target_language, todo, checks=fingerprint)
                for text, (reason, first_reason, output) in known.items():
                    out.problems.append(Problem(tuple(where[text]), text, reason, first_reason, output))
                out.known_failures = len(known)
                todo = [t for t in todo if t not in known]
        out.translated = len(todo)

        failed, done = [], 0     # failed: (text, reason of the first attempt)
        for batch in make_batches(todo, self.batch_size, self.max_batch_chars):
            good = {}
            for text, result, reason in self._attempt(batch, source_language, target_language, 0):
                if reason is None:
                    good[text] = result
                    store(text, result)
                else:
                    failed.append((text, reason))
            # a translation that may be cut off is not remembered: it is warned about on every run
            good = {t: r for t, r in good.items() if t not in self._cut}
            if cache is not None and good:
                cache.put_many(ident, source_language, target_language, good)
            done += len(batch)
            if progress:
                progress(done - len(failed), len(todo))

        out.retried = len(failed)
        for n, (text, first_reason) in enumerate(failed):
            (_, result, reason), = self._attempt([text], source_language, target_language, 1)
            if reason is None:
                store(text, result)
                if cache is not None and text not in self._cut:
                    cache.put_many(ident, source_language, target_language, {text: result})
            else:
                out.problems.append(Problem(tuple(where[text]), text, reason, first_reason, result))
                if cache is not None:
                    cache.put_failure(ident, source_language, target_language, text, reason, first_reason, result,
                                      checks=fingerprint)
            if progress:
                progress(len(todo) - len(failed) + n + 1, len(todo))
        out.truncated = sorted(((min(where[t]), t) for t in self._cut))
        out.truncated = [(i + 1, t) for i, t in out.truncated]
        if out.truncated:
            warnings.warn(f"{self.name}: {len(out.truncated)} translation(s) stopped at the limit on new tokens "
                          f"(max_new_tokens) and may be cut off, first at line {out.truncated[0][0]}; "
                          f"raise max_new_tokens or split the sentence", TranslationWarning, stacklevel=2)
        if out.problems:
            out.problems.sort(key=lambda p: p.lines)
            warnings.warn(f"{self.name}: {len(out.problems)} sentence(s) could not be translated "
                          f"({_reasons(out.problems)}); they get an empty translation", TranslationWarning,
                          stacklevel=2)
        return out

    def _attempt(self, batch, source_language, target_language, attempt):
        """-> [(text, output, reason or None)]"""
        self.last_truncated = ()
        results = self.translate_batch(batch, source_language, target_language, attempt)
        cut, self.last_truncated = set(self.last_truncated or ()), ()
        if results is None or len(results) != len(batch):
            return [(t, "", COUNT) for t in batch]
        triples = []
        for n, (text, raw) in enumerate(zip(batch, results)):
            raw = "" if raw is None else str(raw)
            reason = self.checks.problem(text, raw)
            if attempt:                          # the second attempt decides: forget the first one's finding
                self._cut.discard(text)
            if reason is None and n in cut:
                self._cut.add(text)
            triples.append((text, raw.strip(), reason))
        return triples


class TranslationWarning(UserWarning):
    pass


def _reasons(problems) -> str:
    counts: dict = {}
    for p in problems:
        counts[p.reason] = counts.get(p.reason, 0) + 1
    return ", ".join(f"{r}: {n}" for r, n in sorted(counts.items()))


def describe_problems(problems: Sequence[Problem], limit: int = 5) -> str:
    """A few lines for the user: which sentences were left empty and why."""
    lines = []
    for p in problems[:limit]:
        src = p.source if len(p.source) <= 70 else p.source[:67] + "..."
        got = p.output if len(p.output) <= 70 else p.output[:67] + "..."
        lines.append(f"  line {p.lines[0] + 1}: {p.reason} (first attempt: {p.first_reason}): {src!r} -> {got!r}")
    if len(problems) > limit:
        lines.append(f"  ... and {len(problems) - limit} more")
    return "\n".join(lines)
