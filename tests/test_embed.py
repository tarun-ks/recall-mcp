"""Tests for ``recall.embed``.

Light unit tests only — anything that loads the actual model lives in
``test_eval.py`` under ``@pytest.mark.embed`` (the heavy lane).
"""

from __future__ import annotations

from recall.embed import DEFAULT_MODEL


def test_default_model_is_bge_small() -> None:
    """The default model is the one all benchmarks and CLAUDE.md docs reference.
    Changing this is a major version bump."""
    assert DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"
