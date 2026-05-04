"""Embedding model wrapper.

Public API frozen at Commit 2.5; internals rewritten in Commit 2.7
behavior-preservingly (MPS warmup at construction time, explicit
batch-size control, DEBUG-level encode logging). The 2.7 commit added
the optional ``batch_size`` kwarg to ``__init__`` under CLAUDE.md §4a's
"Frozen API extension policy" rule (optional, default-preserving,
documented at landing).

Frozen public surface:

    Embedder(model_name: str = "BAAI/bge-small-en-v1.5",
             model_revision: str | None = None,
             cache_folder: Path | None = None,
             batch_size: int | None = None)   # added 2.7

    Embedder.encode(texts: Sequence[str]) -> np.ndarray
        # shape: (len(texts), dim); L2-normalized

    Embedder.dim          # embedding dimension (read-only)
    Embedder.model_name   # the model id passed at construction (read-only)
    Embedder.model_revision  # the revision pinned, or None (read-only)

Anything else (private methods, internal caching, batching strategy) is
implementation detail and may change.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE = Path.home() / ".recall" / "models"

# Default batch size for sentence-transformers' internal batching. 128 was
# validated against M-series MPS and CI Linux CPU at 2.7 (CLAUDE.md §4a
# "Performance contract"). Empirically: batch=128 ran ~1s faster than
# batch=64 on M-series (24.40s vs 25.49s median across 3-run samples) on
# nl2bash; encode-bound stages benefit from the larger batch grouping.
# Memory math: 128 × 512-token max × 384-dim × 4 bytes ≈ 100 MB activation
# headroom on a 7 GB CI runner — two orders of magnitude under budget.
# Override with ``RECALL_EMBED_BATCH_SIZE`` env var or ``batch_size`` kwarg.
DEFAULT_BATCH_SIZE = 128

_LOG = logging.getLogger(__name__)


class Embedder:
    """sentence-transformers wrapper with MPS warmup + explicit batching."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_revision: str | None = None,
        cache_folder: Path | None = None,
        batch_size: int | None = None,
    ) -> None:
        # Lazy import: sentence_transformers pulls torch (~25s + ~150 MB),
        # which we don't want triggered just by importing `recall.embed`
        # (e.g. during pytest collection of an unrelated test file).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_revision = model_revision

        # Resolve batch_size: explicit kwarg > env var > default.
        if batch_size is not None:
            self._batch_size = batch_size
        else:
            env_val = os.environ.get("RECALL_EMBED_BATCH_SIZE")
            self._batch_size = int(env_val) if env_val else DEFAULT_BATCH_SIZE

        cache = cache_folder if cache_folder is not None else DEFAULT_CACHE
        cache.mkdir(parents=True, exist_ok=True)
        self._model: SentenceTransformer = SentenceTransformer(
            model_name,
            revision=model_revision,
            cache_folder=str(cache),
        )
        # sentence-transformers 5.x renamed get_sentence_embedding_dimension →
        # get_embedding_dimension. getattr keeps us compatible across both.
        get_dim = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        d = get_dim()
        if d is None:  # defensive — happens only with hand-built models
            raise RuntimeError(f"could not determine embedding dim for {model_name!r}")
        self.dim: int = d

        # MPS warmup: first encode on Apple Silicon pays a ~14s kernel-
        # compilation cost. Doing a tiny dummy encode at construction
        # amortizes that into init time — by the time the first real
        # encode happens, MPS kernels are compiled. CI Linux CPU has no
        # warmup cost; the call is a fast no-op there.
        try:
            self._model.encode(
                ["warmup"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self._batch_size,
            )
        except BaseException as e:
            # Don't brick Embedder construction on a warmup failure — the
            # first real encode just pays the cost itself, identical to
            # pre-2.7 behavior. Future PyTorch versions or unusual
            # hardware shouldn't break us here.
            _LOG.debug("MPS warmup failed (non-fatal): %r", e)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an (N, dim) float32 array, L2-normalized.

        Normalization on output means cosine similarity reduces to dot
        product downstream — sqlite-vec uses L2 distance by default but
        with normalized vectors L2 and cosine rank identically, so the
        same vectors work for both.
        """
        _LOG.debug("encode: %d texts, batch_size=%d", len(texts), self._batch_size)
        out: np.ndarray = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self._batch_size,
        )
        # sentence-transformers may return float64 on some platforms; coerce
        # to float32 to match sqlite-vec's FLOAT[N] storage.
        return np.asarray(out, dtype=np.float32)


__all__ = ("DEFAULT_BATCH_SIZE", "DEFAULT_MODEL", "Embedder")
