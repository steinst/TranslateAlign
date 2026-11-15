"""Encoder-decoder models from Hugging Face: Opus-MT (Marian), NLLB-200, MADLAD-400.

Each family tells the model the languages in its own way (docs/MODELS.md):
  opus-mt   one model per language pair; the model id may contain {src} and {trg}
            (`Helsinki-NLP/opus-mt-{src}-{trg}`); models with several target languages get `>>fra<< ` in front.
  nllb      source language set on the tokenizer, target language as the first decoder token (`fra_Latn`).
  madlad    `<2fr> ` in front of the sentence.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence

from . import _hf
from .base import Checks, Translator
from .definition import ModelDefinition
from .languages import FAMILIES, family_of, madlad_token, nllb_code, opus_prefix

# Second attempt: greedy search, repetitions made expensive (loops and empty beams are the usual failures).
DEFAULT_RETRY = {"num_beams": 1, "do_sample": False, "repetition_penalty": 1.3, "no_repeat_ngram_size": 4}


class Seq2SeqTranslator(Translator):
    def __init__(self, model: str, family: Optional[str] = None, *, languages: Optional[Mapping] = None,
                 generation: Optional[Mapping] = None, retry: Optional[Mapping] = None,
                 load: Optional[Mapping] = None, checks: Optional[Checks] = None, device: Optional[str] = None,
                 trust_remote_code: bool = False, needs_remote_code: bool = False, name: Optional[str] = None):
        self.definition = ModelDefinition(
            name=name or model, model=model, backend="seq2seq", family=family or family_of(model),
            languages=dict(languages or {}), generation=dict(generation or {}),
            retry=dict(DEFAULT_RETRY if retry is None else retry), load=dict(load or {}),
            trust_remote_code=needs_remote_code)
        if self.definition.family not in FAMILIES:
            raise ValueError(f"cannot tell the model family of '{model}': give family = one of "
                             f"{', '.join(FAMILIES)}")
        self.name = self.definition.name
        self.family = self.definition.family
        self.batch_size = self.definition.setting("batch_size")
        self.max_batch_chars = self.definition.setting("max_batch_chars")
        if checks is not None:
            self.checks = checks
        self._device = device or self.definition.load.get("device")
        self._remote = _hf.check_remote_code(needs_remote_code, trust_remote_code, model)
        self._loaded = None          # (model id, tokenizer, model)

    @classmethod
    def from_definition(cls, d: ModelDefinition, *, device=None, trust_remote_code=False, checks=None):
        return cls(d.model, d.family, languages=d.languages, generation=d.generation,
                   retry=d.retry or None, load=d.load, checks=checks, device=device,
                   trust_remote_code=trust_remote_code, needs_remote_code=d.trust_remote_code, name=d.name)

    def model_id(self, source_language: str, target_language: str) -> str:
        return self.definition.model.replace("{src}", source_language).replace("{trg}", target_language)

    def cache_id(self, source_language, target_language):
        return f"{self.model_id(source_language, target_language)}#{self.definition.fingerprint()}"

    # -- the model ---------------------------------------------------------------------------------------
    def _load(self, source_language, target_language):
        model_id = self.model_id(source_language, target_language)
        if self._loaded and self._loaded[0] == model_id:
            return self._loaded[1:]
        self.close()
        _, transformers = _hf.imports()
        device = _hf.pick_device(self._device)
        extra = {"trust_remote_code": True} if self._remote else {}
        if self.definition.load.get("revision"):
            extra["revision"] = self.definition.load["revision"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **extra)
        model = _hf.load_model(transformers.AutoModelForSeq2SeqLM, model_id, self.definition.load, device,
                               self._remote)
        self._loaded = (model_id, tokenizer, model)
        return tokenizer, model

    def close(self):
        if self._loaded:
            self._loaded = None
            _hf.free()

    # -- languages ---------------------------------------------------------------------------------------
    def prepare(self, tokenizer, sentences: Sequence[str], source_language: str, target_language: str):
        """-> (the texts for the tokenizer, extra arguments for generate()); sets the tokenizer's source
        language where the family has one."""
        over = self.definition.languages
        if self.family == "opus-mt":
            prefix = opus_prefix(target_language, getattr(tokenizer, "supported_language_codes", None), over)
            return [prefix + s for s in sentences], {}
        unk = tokenizer.unk_token_id
        if self.family == "madlad":
            token = madlad_token(target_language, lambda t: tokenizer.convert_tokens_to_ids(t) != unk, over)
            return [f"{token} {s}" for s in sentences], {}
        # transformers 5 has no `additional_special_tokens` any more: `all_special_tokens` holds the codes
        special = (getattr(tokenizer, "additional_special_tokens", None)
                   or getattr(tokenizer, "all_special_tokens", None) or [])
        known = [t for t in special if re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", t)]
        src, trg = nllb_code(source_language, known, over), nllb_code(target_language, known, over)
        for code in (src, trg):
            if tokenizer.convert_tokens_to_ids(code) == unk:
                raise ValueError(f"the model has no language code '{code}'")
        tokenizer.src_lang = src
        return list(sentences), {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(trg)}

    # -- translation -------------------------------------------------------------------------------------
    def translate_batch(self, sentences, source_language, target_language, attempt=0):
        torch, _ = _hf.imports()
        tokenizer, model = self._load(source_language, target_language)
        d = self.definition
        texts, extra = self.prepare(tokenizer, sentences, source_language, target_language)
        limit = d.setting("max_input_tokens", attempt) or tokenizer.model_max_length
        limit = min(limit, getattr(model.config, "max_position_embeddings", None) or limit)
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=limit if limit < 100000 else None).to(model.device)
        kwargs = d.generate_kwargs(attempt)
        kwargs.update(extra)
        kwargs["max_new_tokens"] = max(8, min(
            d.setting("max_new_tokens", attempt),
            int(d.setting("new_tokens_per_input_token", attempt) * enc["input_ids"].shape[1]) + 16))
        kwargs.setdefault("max_length", None)       # max_new_tokens decides, not the model's default
        with torch.inference_mode():
            out = model.generate(**enc, **kwargs)
        self.last_truncated = _hf.hit_limit(out.tolist(), _hf.finish_ids(model, tokenizer))
        return tokenizer.batch_decode(out, skip_special_tokens=True)
