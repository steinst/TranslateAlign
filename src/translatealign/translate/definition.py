"""Model definition files (TOML): everything the program must know to translate with one model.

The format is described in docs/MODELS.md. Unknown keys are an error, so that a misspelt setting is noticed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib

BACKENDS = ("chat", "seq2seq")
PROMPT_MODES = ("chat", "plain")
LINE_RULES = ("keep", "first", "last", "join")
QUANTIZATIONS = ("none", "4bit", "8bit")
PLACEHOLDERS = ("{source_language}", "{target_language}", "{source_code}", "{target_code}", "{text}")

# [generation] keys that are ours; every other key is passed to the model's generate().
OWN_GENERATION = {"batch_size": 16, "max_batch_chars": 8000, "max_new_tokens": 256,
                  "new_tokens_per_input_token": 3.0, "max_input_tokens": 0, "stop": []}

_TOP = {"name", "backend", "model", "family", "licence", "license", "description", "citation",
        "trust_remote_code", "languages", "prompt", "generation", "retry", "cleanup", "load", "checks"}
_PROMPT = {"mode", "system", "template", "content", "response_prefix", "template_args"}
_CLEANUP = {"remove", "strip_prefixes", "lines", "strip_quotes"}
_LOAD = {"dtype", "quantization", "device", "model_class", "subfolder", "revision"}
_CHECKS = {"min_ratio", "max_ratio", "slack", "leak_patterns", "extra_leak_patterns"}


class DefinitionError(ValueError):
    pass


@dataclass
class ModelDefinition:
    name: str
    model: str                                   # Hugging Face id or a local directory
    backend: str = "chat"
    family: Optional[str] = None                 # seq2seq: opus-mt, nllb, madlad
    licence: str = ""
    description: str = ""
    citation: str = ""
    trust_remote_code: bool = False              # the model needs code from its repository
    languages: dict = field(default_factory=dict)    # chat: code -> name in the prompt; seq2seq: code -> model's code
    prompt: dict = field(default_factory=dict)
    generation: dict = field(default_factory=dict)
    retry: dict = field(default_factory=dict)        # [generation] overrides for the second attempt
    cleanup: dict = field(default_factory=dict)
    load: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.backend not in BACKENDS:
            raise DefinitionError(f"backend must be one of {', '.join(BACKENDS)}, not '{self.backend}'")
        _only(self.prompt, _PROMPT, "prompt")
        _only(self.cleanup, _CLEANUP, "cleanup")
        _only(self.load, _LOAD, "load")
        _only(self.checks, _CHECKS, "checks")
        if self.backend == "chat":
            mode = self.prompt.get("mode", "chat")
            if mode not in PROMPT_MODES:
                raise DefinitionError(f"[prompt] mode must be one of {', '.join(PROMPT_MODES)}")
            content, template = self.prompt.get("content"), self.prompt.get("template")
            if (content is None) == (template is None):
                raise DefinitionError("[prompt] needs either `template` or a [prompt.content] table")
            if content is not None and mode != "chat":
                raise DefinitionError("[prompt.content] needs mode = \"chat\"")
            carrier = template if template is not None else json.dumps(content)
            if "{text}" not in carrier:
                raise DefinitionError("the prompt has no {text}")
        elif self.prompt:
            raise DefinitionError("[prompt] is for backend = \"chat\"")
        if self.cleanup.get("lines", "keep") not in LINE_RULES:
            raise DefinitionError(f"[cleanup] lines must be one of {', '.join(LINE_RULES)}")
        if self.load.get("quantization", "none") not in QUANTIZATIONS:
            raise DefinitionError(f"[load] quantization must be one of {', '.join(QUANTIZATIONS)}")
        stop = self.generation.get("stop", [])
        if not isinstance(stop, list) or not all(isinstance(s, str) and s for s in stop):
            raise DefinitionError("[generation] stop must be a list of non-empty strings")

    def setting(self, key: str, attempt: int = 0):
        """One of our own [generation] settings, with the [retry] override at the second attempt."""
        if attempt and key in self.retry:
            return self.retry[key]
        return self.generation.get(key, OWN_GENERATION[key])

    def generate_kwargs(self, attempt: int = 0) -> dict:
        """The settings passed to the model's generate()."""
        merged = dict(self.generation)
        if attempt:
            merged.update(self.retry)
        return {k: v for k, v in merged.items() if k not in OWN_GENERATION}

    def fingerprint(self) -> str:
        """Changes whenever something changes that can change the translation."""
        d = asdict(self)
        for k in ("name", "licence", "description", "citation", "checks"):
            d.pop(k)
        d["load"] = {k: v for k, v in self.load.items() if k != "device"}
        d["generation"] = {k: v for k, v in self.generation.items() if k not in ("batch_size", "max_batch_chars")}
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def _only(table: Mapping, allowed, where: str) -> None:
    if not isinstance(table, Mapping):
        raise DefinitionError(f"[{where}] must be a table")
    unknown = sorted(set(table) - set(allowed))
    if unknown:
        raise DefinitionError(f"unknown key(s) in {'[' + where + ']' if where else 'the definition'}: "
                              f"{', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})")


def definition_from_dict(data: Mapping[str, Any], default_name: str = "") -> ModelDefinition:
    data = dict(data)
    _only(data, _TOP, "")
    if "license" in data:
        data.setdefault("licence", data.pop("license"))
        data.pop("license", None)
    if "model" not in data:
        raise DefinitionError("the definition has no `model` (a Hugging Face id or a directory)")
    data.setdefault("name", default_name or data["model"])
    return ModelDefinition(**data)


def load_definition(path: str) -> ModelDefinition:
    import os
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise DefinitionError(f"{path}: not valid TOML: {e}") from None
    try:
        return definition_from_dict(data, os.path.splitext(os.path.basename(path))[0])
    except DefinitionError as e:
        raise DefinitionError(f"{path}: {e}") from None
