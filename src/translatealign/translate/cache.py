"""A cache of translations on disk (one SQLite file), keyed by model, languages and text.

Sentences that failed the checks twice are remembered too (table `failures`), so that the model is not loaded
again on every run for sentences it cannot translate; `retry_failed` of `Translator.translate` tries them again.

Every batch is committed when it is written, so a run that is interrupted loses at most the batch in work.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

ENV = "TRANSLATEALIGN_CACHE"


def default_cache_path() -> str:
    """$TRANSLATEALIGN_CACHE, else translatealign/translations.sqlite in the user's cache directory."""
    if os.environ.get(ENV):
        return os.environ[ENV]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(Path.home(), ".cache")
    return os.path.join(base, "translatealign", "translations.sqlite")


def _key(model: str, source_language: str, target_language: str, text: str) -> str:
    h = hashlib.sha256()
    for part in (model, source_language, target_language, text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class TranslationCache:
    def __init__(self, path: Optional[str] = None):
        self.path = str(path) if path else default_cache_path()
        if os.path.dirname(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._db = sqlite3.connect(self.path, timeout=60)
        self._db.execute("CREATE TABLE IF NOT EXISTS translations (key TEXT PRIMARY KEY, model TEXT, "
                         "source_language TEXT, target_language TEXT, source TEXT, translation TEXT, "
                         "created REAL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS failures (key TEXT PRIMARY KEY, model TEXT, "
                         "source_language TEXT, target_language TEXT, source TEXT, reason TEXT, "
                         "first_reason TEXT, output TEXT, created REAL)")
        columns = [row[1] for row in self._db.execute("PRAGMA table_info(failures)")]
        if "checks" not in columns:        # an older cache: no fingerprint = unknown = try again
            self._db.execute("ALTER TABLE failures ADD COLUMN checks TEXT")
        self._db.commit()

    def _select(self, columns, table, model, source_language, target_language, texts) -> dict:
        keys = {_key(model, source_language, target_language, t): t for t in texts}
        found, names = {}, list(keys)
        for i in range(0, len(names), 500):
            part = names[i:i + 500]
            rows = self._db.execute(f"SELECT key, {columns} FROM {table} WHERE key IN "
                                    f"({','.join('?' * len(part))})", part)
            for key, *values in rows:
                found[keys[key]] = values[0] if len(values) == 1 else tuple(values)
        return found

    def get_many(self, model: str, source_language: str, target_language: str, texts: Sequence[str]) -> dict:
        """-> {text: translation} for the texts that are in the cache."""
        return self._select("translation", "translations", model, source_language, target_language, texts)

    def get_failures(self, model: str, source_language: str, target_language: str,
                     texts: Sequence[str], checks: Optional[str] = None) -> dict:
        """-> {text: (reason, reason of the first attempt, output)} for the texts known to fail under the checks
        with this fingerprint (`Checks.fingerprint()`); a failure recorded under other or unknown checks does not
        count. checks=None: all failures."""
        found = self._select("reason, first_reason, output, checks", "failures", model, source_language,
                             target_language, texts)
        return {t: v[:3] for t, v in found.items() if checks is None or v[3] == checks}

    def put_failure(self, model: str, source_language: str, target_language: str, text: str, reason: str,
                    first_reason: str, output: str, checks: Optional[str] = None) -> None:
        key = _key(model, source_language, target_language, text)
        self._db.execute("DELETE FROM translations WHERE key = ?", (key,))     # a stale success, if any
        self._db.execute("INSERT OR REPLACE INTO failures (key, model, source_language, target_language, source, "
                         "reason, first_reason, output, created, checks) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         (key, model, source_language, target_language, text, reason, first_reason, output,
                          time.time(), checks))
        self._db.commit()

    def put_many(self, model: str, source_language: str, target_language: str,
                 translations: Mapping[str, str]) -> None:
        now = time.time()
        self._db.executemany("DELETE FROM failures WHERE key = ?",
                             [(_key(model, source_language, target_language, text),) for text in translations])
        self._db.executemany(
            "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(_key(model, source_language, target_language, text), model, source_language, target_language,
              text, translation, now) for text, translation in translations.items()])
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM translations").fetchone()[0]

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
