"""Decoder ("chat") models, driven by a model definition file (docs/MODELS.md).

The definition gives the prompt, the names of the languages, the generation settings, the stop strings and the
rules that clean the answer. One sentence is one prompt; the answer must be one line.
"""
from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from . import _hf
from .base import DEFAULT_LEAK_PATTERNS, Checks, Translator
from .definition import PLACEHOLDERS, ModelDefinition
from .languages import language_name

_QUOTES = [('"', '"'), ("'", "'"), ("“", "”"), ("„", "“"), ("„", "”"), ("«", "»"), ("»", "«"), ("‘", "’"),
           ("「", "」"), ("`", "`")]
_QUOTE_CHARS = {c for pair in _QUOTES for c in pair}


def fill(template: str, values: dict) -> str:
    """Put the values in place of the placeholders (no str.format: the text may contain braces)."""
    return re.sub("|".join(re.escape(p) for p in PLACEHOLDERS), lambda m: values[m.group(0)], template)


def clean(raw: str, source: str, cleanup: dict, stop: Sequence[str] = ()) -> str:
    """The clean-up rules of a definition, in this order: cut at the first stop string; remove the `remove`
    patterns; strip the `strip_prefixes`; the `lines` rule; `strip_quotes`."""
    text = raw
    # a stop string at the very beginning (a model that opens with a newline) would leave nothing: skip it
    lead = len(text) - len(text.lstrip())
    cuts = [text.find(s, lead) for s in stop]
    cuts = [c for c in cuts if c >= 0]
    if cuts:
        text = text[:min(cuts)]
    for pattern in cleanup.get("remove", ()):
        text = re.sub(pattern, "", text, flags=re.DOTALL)
    text = text.strip()
    prefixes = sorted(cleanup.get("strip_prefixes", ()), key=len, reverse=True)
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if p and text.lower().startswith(p.lower()):
                text, changed = text[len(p):].lstrip(), True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rule = cleanup.get("lines", "keep")
    if lines and rule == "first":
        text = lines[0]
    elif lines and rule == "last":
        text = lines[-1]
    elif rule == "join":
        text = " ".join(lines)
    if cleanup.get("strip_quotes", False) and len(text) >= 2 and source.strip()[:1] not in _QUOTE_CHARS:
        for a, b in _QUOTES:
            if text.startswith(a) and text.endswith(b) and a not in text[1:-1] and b not in text[1:-1]:
                text = text[1:-1].strip()
                break
    return text


def prompt_fragments(definition: ModelDefinition, min_length: int = 15) -> list:
    """The fixed pieces of the prompt; if one of them shows up in an answer, the prompt has leaked."""
    pieces = []
    for part in (definition.prompt.get("template"), definition.prompt.get("system")):
        if part:
            for piece in re.split("|".join(re.escape(p) for p in PLACEHOLDERS) + r"|\n", part):
                if len(piece.strip()) >= min_length:
                    pieces.append(piece.strip())
    return pieces


def checks_from(definition: ModelDefinition) -> Checks:
    c = definition.checks
    patterns = tuple(c.get("leak_patterns", DEFAULT_LEAK_PATTERNS)) + tuple(c.get("extra_leak_patterns", ()))
    kwargs = {k: c[k] for k in ("min_ratio", "max_ratio", "slack") if k in c}
    return Checks(leak_patterns=patterns, prompt_fragments=tuple(
        prompt_fragments(definition) if definition.backend == "chat" else ()), **kwargs)


class ChatTranslator(Translator):
    """generate: for tests and other runtimes, a function (prompts, attempt) -> answers that replaces the
    Hugging Face model. A prompt is a list of chat messages (mode "chat") or a string (mode "plain")."""

    def __init__(self, definition: ModelDefinition, *, device: Optional[str] = None,
                 trust_remote_code: bool = False, checks: Optional[Checks] = None,
                 generate: Optional[Callable] = None):
        if definition.backend != "chat":
            raise ValueError(f"the definition '{definition.name}' is for the backend {definition.backend}")
        self.definition = definition
        self.name = definition.name
        self.batch_size = definition.setting("batch_size")
        self.max_batch_chars = definition.setting("max_batch_chars")
        self.checks = checks or checks_from(definition)
        self._device = device or definition.load.get("device")
        self._generate = generate
        self._remote = False if generate else _hf.check_remote_code(
            definition.trust_remote_code, trust_remote_code, definition.model)
        self._loaded = None

    def cache_id(self, source_language, target_language):
        return f"{self.definition.model}#{self.definition.fingerprint()}"

    # -- prompts -----------------------------------------------------------------------------------------
    def build_prompts(self, sentences: Sequence[str], source_language: str, target_language: str) -> list:
        p = self.definition.prompt
        carrier = (p.get("template") or "") + (p.get("system") or "") + str(p.get("content") or "")
        values = {"{source_code}": source_language, "{target_code}": target_language}
        for key, code in (("{source_language}", source_language), ("{target_language}", target_language)):
            values[key] = language_name(code, self.definition.languages) if key in carrier else ""
        prompts = []
        for s in sentences:
            v = dict(values)
            v["{text}"] = s
            if p.get("mode", "chat") == "plain":
                prompts.append(fill(p["template"], v))
                continue
            if p.get("content") is not None:
                content = [{k: fill(x, v) if isinstance(x, str) else x for k, x in p["content"].items()}]
            else:
                content = fill(p["template"], v)
            messages = [{"role": "system", "content": fill(p["system"], v)}] if p.get("system") else []
            prompts.append(messages + [{"role": "user", "content": content}])
        return prompts

    # -- translation -------------------------------------------------------------------------------------
    def translate_batch(self, sentences, source_language, target_language, attempt=0):
        self.last_truncated = ()
        prompts = self.build_prompts(sentences, source_language, target_language)
        generate = self._generate or self._hf_generate
        answers = generate(prompts, attempt)
        if answers is None or len(answers) != len(sentences):
            return answers
        d = self.definition
        return [clean(a, s, d.cleanup, d.setting("stop", attempt)) for a, s in zip(answers, sentences)]

    def _load(self):
        if self._loaded:
            return self._loaded
        _, transformers = _hf.imports()
        d = self.definition
        device = _hf.pick_device(self._device)
        extra = {"trust_remote_code": True} if self._remote else {}
        if d.load.get("subfolder"):
            extra["subfolder"] = d.load["subfolder"]
        if d.load.get("revision"):
            extra["revision"] = d.load["revision"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(d.model, **extra)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_class = getattr(transformers, d.load.get("model_class", "AutoModelForCausalLM"))
        model = _hf.load_model(model_class, d.model, d.load, device, self._remote)
        self._loaded = (tokenizer, model)
        return self._loaded

    def close(self):
        if self._loaded:
            self._loaded = None
            _hf.free()

    def render(self, tokenizer, prompt) -> str:
        """The text given to the model for one prompt."""
        p = self.definition.prompt
        if isinstance(prompt, str):
            return prompt + p.get("response_prefix", "")
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True,
                                             **p.get("template_args", {}))
        return text + p.get("response_prefix", "")

    def _hf_generate(self, prompts, attempt):
        torch, _ = _hf.imports()
        tokenizer, model = self._load()
        d = self.definition
        plain = d.prompt.get("mode", "chat") == "plain"
        texts = [self.render(tokenizer, p) for p in prompts]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=plain).to(model.device)
        sentences = [p if plain else _text_of(p) for p in prompts]
        longest = max(len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sentences)
        kwargs = {"do_sample": False}
        kwargs.update(d.generate_kwargs(attempt))
        kwargs["max_new_tokens"] = max(8, min(
            d.setting("max_new_tokens", attempt),
            int(d.setting("new_tokens_per_input_token", attempt) * longest) + 16))
        kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
        stop = d.setting("stop", attempt)
        if stop:
            kwargs.update(stop_strings=list(stop), tokenizer=tokenizer)
        kwargs.setdefault("max_length", None)       # max_new_tokens decides, not the model's default
        with torch.inference_mode():
            out = model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new, skip_special_tokens=True)
        self.last_truncated = _hf.hit_limit(new.tolist(), _hf.finish_ids(model, tokenizer), texts, stop or ())
        return texts


def _text_of(messages) -> str:
    content = messages[-1]["content"]
    return content if isinstance(content, str) else str(content[0].get("text", ""))
