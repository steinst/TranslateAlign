"""Translators: sentences in, the same number of sentences out.

    translator = load_translator(model="Helsinki-NLP/opus-mt-de-fr")          # an encoder-decoder model
    translator = load_translator(model_file="my_model.toml")                  # a model definition (docs/MODELS.md)
    with TranslationCache() as cache:
        french = translator.translate(german_sentences, "de", "fr", cache=cache, keep=[".EOA"])
    french.problems      # sentences left empty because the model's output failed the checks twice
"""
from __future__ import annotations

from typing import Optional

from .base import Checks, Problem, Translations, TranslationWarning, Translator, describe_problems, make_batches
from .cache import TranslationCache, default_cache_path
from .chat import ChatTranslator, checks_from
from .definition import DefinitionError, ModelDefinition, definition_from_dict, load_definition
from .file import FileTranslator
from .languages import UnknownLanguage
from .seq2seq import Seq2SeqTranslator

__all__ = ["Translator", "Translations", "Problem", "Checks", "TranslationWarning", "TranslationCache",
           "default_cache_path", "FileTranslator", "Seq2SeqTranslator", "ChatTranslator", "ModelDefinition",
           "DefinitionError", "UnknownLanguage", "load_definition", "definition_from_dict", "load_translator",
           "translator_from_definition", "describe_problems", "make_batches"]


def translator_from_definition(definition: ModelDefinition, *, device: Optional[str] = None,
                               trust_remote_code: bool = False) -> Translator:
    if definition.backend == "seq2seq":
        return Seq2SeqTranslator.from_definition(definition, device=device, trust_remote_code=trust_remote_code,
                                                 checks=checks_from(definition))
    return ChatTranslator(definition, device=device, trust_remote_code=trust_remote_code)


def load_translator(model: Optional[str] = None, model_file: Optional[str] = None, *,
                    device: Optional[str] = None, trust_remote_code: bool = False) -> Translator:
    """model: the Hugging Face id (or directory) of an Opus-MT, NLLB-200 or MADLAD-400 model.
    model_file: a model definition file. The model itself is loaded at the first translation."""
    if (model is None) == (model_file is None):
        raise ValueError("give either a model or a model definition file")
    if model_file is not None:
        return translator_from_definition(load_definition(model_file), device=device,
                                          trust_remote_code=trust_remote_code)
    shipped = shipped_definition(model)
    if shipped:
        return translator_from_definition(load_definition(shipped), device=device,
                                          trust_remote_code=trust_remote_code)
    return Seq2SeqTranslator(model, device=device, trust_remote_code=trust_remote_code)


def shipped_definition(name: str) -> Optional[str]:
    """The path of the definition that ships with the package under this name (`nllb-200`), or None."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", name + ".toml")
    return path if os.path.isfile(path) else None
