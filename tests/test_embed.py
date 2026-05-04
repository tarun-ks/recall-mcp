"""Tests for ``recall.embed``.

Light unit tests are unmarked (no model load). Anything loading the
actual sentence-transformers model is ``@pytest.mark.embed`` (heavy
lane); ``test_eval.py`` covers the end-to-end semantic path,
``test_embed_behavior_preservation.py`` (added 2.7) is the recall@5
behavior gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from recall.embed import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, Embedder


def test_default_model_is_bge_small() -> None:
    """The default model is the one all benchmarks and CLAUDE.md docs reference.
    Changing this is a major version bump."""
    assert DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"


def test_default_batch_size_is_128() -> None:
    """The 2.7 default batch size is 128 (validated against M-series MPS and
    CI Linux CPU; CLAUDE.md §4a "Performance contract"). Changing this
    requires re-running the full eval gate to confirm behavior preservation."""
    assert DEFAULT_BATCH_SIZE == 128


@pytest.mark.embed
def test_embedder_default_batch_size() -> None:
    """Embedder() with no kwargs uses DEFAULT_BATCH_SIZE."""
    e = Embedder()
    assert e._batch_size == DEFAULT_BATCH_SIZE


@pytest.mark.embed
def test_embedder_env_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """RECALL_EMBED_BATCH_SIZE env var overrides the default."""
    monkeypatch.setenv("RECALL_EMBED_BATCH_SIZE", "8")
    e = Embedder()
    assert e._batch_size == 8


@pytest.mark.embed
def test_embedder_kwarg_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit ``batch_size`` kwarg takes precedence over env."""
    monkeypatch.setenv("RECALL_EMBED_BATCH_SIZE", "8")
    e = Embedder(batch_size=16)
    assert e._batch_size == 16


@pytest.mark.embed
def test_encode_outputs_l2_normalized() -> None:
    """``encode()`` returns L2-normalized vectors so cosine == dot product
    downstream. sqlite-vec's L2 distance and cosine rank identically on
    normalized vectors; this invariant is load-bearing."""
    e = Embedder()
    out = e.encode(["hello world", "another text", "third one"])
    assert out.shape == (3, e.dim)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), atol=1e-5)
