from .bleu import BleuScorer, tokenize, tokenize_13a
from .chrf import ChrfScorer

__all__ = ["BleuScorer", "ChrfScorer", "tokenize", "tokenize_13a", "join"]


def join(sentences):
    """Join a sequence of sentences into one string (the unit scored for 1-to-n links)."""
    return " ".join(sentences)
