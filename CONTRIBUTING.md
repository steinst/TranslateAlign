# Contributing

Use Python 3.10 or newer. From a clone of this repository:

```sh
python -m venv .venv
# Activate the environment using the command for your shell.
python -m pip install -e ".[dev]"
python -m pytest -q
```

Reference tests need a separate clone of Bleualign. Set
`TRANSLATEALIGN_BLEUALIGN_REPO` to its path; CI uses commit
`5a11b7f8831b5463d579f4fbd6f114194020fb12`. Without it those tests are skipped.
The real-model test is opt-in: install `.[models]` and set
`TRANSLATEALIGN_RUN_MODELS=1` to download and run Opus-MT de-fr.

TranslateAlign implements the published methods independently. Do not copy
Bleualign source code or evaluation texts into this repository. The numeric
reference fixtures have their own licence; see [tests/reference/README.md](tests/reference/README.md).

Build and validate distributions with:

```sh
python -m build
python -m twine check --strict dist/*
```

Tests, reference fixtures, and evaluation tools stay in the GitHub repository;
they are excluded from the installed package and source distribution.

For evaluation tools, set `TRANSLATEALIGN_BLEUALIGN_REPO` and, for ParIce,
`TRANSLATEALIGN_SENTALIGN2` to the corresponding local clones. The latter defaults
to a sibling `SentAlign2` directory. `TRANSLATEALIGN_EVAL7` selects the output
directory (default: `evaluation-output` in the current directory). Pass
`--root /path/to/stored-output` to `tools/sentalign_baseline.py`; that directory
contains `v2/` and `paper/`, each with `<dataset>/output/<document>.txt.path`.
See [docs/RESULTS.md](docs/RESULTS.md) for the reproduction commands.
