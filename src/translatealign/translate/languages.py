"""Language codes and names.

The user gives ISO 639-1 codes (`de`, `fr`, `is`) or a code of the model itself (`deu_Latn`, `>>fra<<`). Every
model family has its own codes; the tables here turn the short code into what the family wants. A code that is
in no table is passed to the model as it is, and a model definition file can override any entry.
"""
from __future__ import annotations

import re
from typing import Iterable, Mapping, Optional

# ISO 639-1 -> (ISO 639-3, English name)
LANGUAGES = {
    "af": ("afr", "Afrikaans"), "am": ("amh", "Amharic"), "ar": ("ara", "Arabic"), "az": ("aze", "Azerbaijani"),
    "be": ("bel", "Belarusian"), "bg": ("bul", "Bulgarian"), "bn": ("ben", "Bengali"), "bs": ("bos", "Bosnian"),
    "ca": ("cat", "Catalan"), "cs": ("ces", "Czech"), "cy": ("cym", "Welsh"), "da": ("dan", "Danish"),
    "de": ("deu", "German"), "el": ("ell", "Greek"), "en": ("eng", "English"), "eo": ("epo", "Esperanto"),
    "es": ("spa", "Spanish"), "et": ("est", "Estonian"), "eu": ("eus", "Basque"), "fa": ("fas", "Persian"),
    "fi": ("fin", "Finnish"), "fo": ("fao", "Faroese"), "fr": ("fra", "French"), "ga": ("gle", "Irish"),
    "gd": ("gla", "Scottish Gaelic"), "gl": ("glg", "Galician"), "gu": ("guj", "Gujarati"),
    "he": ("heb", "Hebrew"), "hi": ("hin", "Hindi"), "hr": ("hrv", "Croatian"), "hu": ("hun", "Hungarian"),
    "hy": ("hye", "Armenian"), "id": ("ind", "Indonesian"), "is": ("isl", "Icelandic"), "it": ("ita", "Italian"),
    "ja": ("jpn", "Japanese"), "ka": ("kat", "Georgian"), "kk": ("kaz", "Kazakh"), "km": ("khm", "Khmer"),
    "kn": ("kan", "Kannada"), "ko": ("kor", "Korean"), "la": ("lat", "Latin"), "lb": ("ltz", "Luxembourgish"),
    "lt": ("lit", "Lithuanian"), "lv": ("lav", "Latvian"), "mk": ("mkd", "Macedonian"), "ml": ("mal", "Malayalam"),
    "mn": ("mon", "Mongolian"), "mr": ("mar", "Marathi"), "ms": ("msa", "Malay"), "mt": ("mlt", "Maltese"),
    "my": ("mya", "Burmese"), "nb": ("nob", "Norwegian Bokmål"), "ne": ("nep", "Nepali"), "nl": ("nld", "Dutch"),
    "nn": ("nno", "Norwegian Nynorsk"), "no": ("nor", "Norwegian"), "pa": ("pan", "Punjabi"),
    "pl": ("pol", "Polish"), "pt": ("por", "Portuguese"), "ro": ("ron", "Romanian"), "ru": ("rus", "Russian"),
    "si": ("sin", "Sinhala"), "sk": ("slk", "Slovak"), "sl": ("slv", "Slovenian"), "sq": ("sqi", "Albanian"),
    "sr": ("srp", "Serbian"), "sv": ("swe", "Swedish"), "sw": ("swa", "Swahili"), "ta": ("tam", "Tamil"),
    "te": ("tel", "Telugu"), "th": ("tha", "Thai"), "tl": ("tgl", "Tagalog"), "tr": ("tur", "Turkish"),
    "uk": ("ukr", "Ukrainian"), "ur": ("urd", "Urdu"), "uz": ("uzb", "Uzbek"), "vi": ("vie", "Vietnamese"),
    "zh": ("zho", "Chinese"),
}

# NLLB-200 (FLORES-200 codes): the cases where the code is not "ISO 639-3 + the only script the model has".
NLLB = {
    "ar": "arb_Arab", "az": "azj_Latn", "fa": "pes_Arab", "lv": "lvs_Latn", "mn": "khk_Cyrl", "ms": "zsm_Latn",
    "nb": "nob_Latn", "ne": "npi_Deva", "no": "nob_Latn", "sq": "als_Latn", "sw": "swh_Latn", "uz": "uzn_Latn",
    "zh": "zho_Hans",
}

FAMILIES = ("opus-mt", "nllb", "madlad")


class UnknownLanguage(ValueError):
    pass


def language_name(code: str, names: Optional[Mapping[str, str]] = None) -> str:
    """The name used in a prompt: from the definition file's table, else the English name."""
    if names and code in names:
        return names[code]
    base = code.replace("_", "-").split("-")[0].lower()
    if names and base in names:
        return names[base]
    if base in LANGUAGES:
        return LANGUAGES[base][1]
    raise UnknownLanguage(f"no name for the language '{code}': add it to [languages] in the model definition")


def iso3(code: str) -> Optional[str]:
    return LANGUAGES[code][0] if code in LANGUAGES else None


def nllb_code(code: str, known: Iterable[str], overrides: Optional[Mapping[str, str]] = None) -> str:
    """`known`: the language codes of the tokenizer (`deu_Latn`, ...)."""
    known = set(known)
    if overrides and code in overrides:
        return overrides[code]
    if code in known:
        return code
    if code in NLLB and (not known or NLLB[code] in known):
        return NLLB[code]
    three = iso3(code) or (code if re.fullmatch(r"[a-z]{3}", code) else None)
    with_script = sorted(k for k in known if three and k.startswith(three + "_"))
    if len(with_script) == 1:
        return with_script[0]
    if len(with_script) > 1:
        raise UnknownLanguage(f"'{code}' is written in several scripts in this model: give one of "
                              f"{', '.join(with_script)}")
    raise UnknownLanguage(f"this model has no language '{code}' (codes look like deu_Latn; give the model's "
                          f"own code, or map it in [languages] of a model definition)")


def madlad_token(code: str, has_token, overrides: Optional[Mapping[str, str]] = None) -> str:
    """The target-language token `<2xx>`; `has_token(token)` says whether the vocabulary has it."""
    if overrides and code in overrides:
        code = overrides[code]
    for c in (code, iso3(code)):
        if c and has_token(f"<2{c}>"):
            return f"<2{c}>"
    raise UnknownLanguage(f"this model has no target language '{code}' (no token <2{code}>)")


def opus_prefix(code: str, supported: Iterable[str], overrides: Optional[Mapping[str, str]] = None) -> str:
    """Opus-MT models with several target languages want `>>xxx<< ` before the sentence; models for one
    pair want nothing. `supported`: the tokenizer's `supported_language_codes` (`>>fra<<`, ...)."""
    supported = list(supported or ())
    if not supported:
        return ""
    if overrides and code in overrides:
        code = overrides[code]
    for c in (code, iso3(code)):
        if c and (f">>{c}<<" in supported or c in supported):
            return c if c.startswith(">>") else f">>{c}<< "
    raise UnknownLanguage(f"this model has no target language '{code}'; it has: {' '.join(sorted(supported))}")


def family_of(model_id: str) -> Optional[str]:
    low = model_id.lower()
    for family, mark in (("opus-mt", "opus-mt"), ("opus-mt", "opus-tatoeba"), ("nllb", "nllb"),
                         ("madlad", "madlad")):
        if mark in low:
            return family
    return None
