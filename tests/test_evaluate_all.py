"""tools/evaluate_all.py must not score fewer documents than it aligned ."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from evalsets import EvalSet  # noqa: E402
from evaluate_all import run  # noqa: E402


def test_evaluation_rejects_missing_gold_document():
    # two documents in the text (separated by .EOA), but only one gold document
    x = ["hello world", ".EOA", "second sentence"]
    s = EvalSet("toy", "en", "fr", x, x, [[((0,), (0,))]], ["first"])
    with pytest.raises(ValueError, match="toy"):
        run(s, x, None, "chrf")
