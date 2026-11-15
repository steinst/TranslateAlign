"""A ready-made translation in a file, behind the translator interface."""
from __future__ import annotations

from typing import Sequence

from .base import EMPTY, Problem, Translations, Translator


class FileTranslator(Translator):
    """The lines of a file are the translation of the sentences, line by line. Nothing is cached or changed;
    an empty line for a non-empty sentence is listed in `.problems` of the result."""

    def __init__(self, path: str):
        self.path = path
        self.name = f"file:{path}"

    def translate(self, sentences: Sequence[str], source_language: str = "", target_language: str = "",
                  **_ignored) -> Translations:
        with open(self.path, encoding="utf-8") as f:
            lines = [line.rstrip("\r\n") for line in f]
        if len(lines) != len(sentences):
            raise ValueError(f"{self.path} has {len(lines)} lines, the text it translates {len(sentences)}")
        out = Translations(lines)
        for i, (s, t) in enumerate(zip(sentences, lines)):
            if not s.strip():
                out.skipped += 1
            elif not t.strip():
                out.problems.append(Problem((i,), s, EMPTY, EMPTY, t))
        return out
