"""TranslateAlign: sentence alignment with machine translation (not the official Bleualign).

    import translatealign
    pairs = translatealign.align(source_sentences, target_sentences, translation=translated_source)
    print(translatealign.cite())
"""

__version__ = "0.1.1"

from .api import Pair, align, cite, read_lines  # noqa: E402  (after __version__: cli imports it)

__all__ = ["align", "cite", "Pair", "read_lines", "__version__"]
