"""Embedding model wrapper.

Public API frozen at Commit 2.5. Commit 2.7 will rewrite internals
(caching, batching, the streaming/batching seam from CLAUDE.md
"Architectural seams") behind this same surface — eval numbers
must match within ±0.01 recall@5 noise band, which is the contract
that makes "behavior-preserving rewrite" testable.

Frozen public surface:

    Embedder(model_name: str = "BAAI/bge-small-en-v1.5",
             model_revision: str | None = None,
             cache_folder: Path | None = None)

    Embedder.encode(texts: Sequence[str]) -> np.ndarray
        # shape: (len(texts), dim); L2-normalized

    Embedder.dim          # embedding dimension (read-only)
    Embedder.model_name   # the model id passed at construction (read-only)
    Embedder.model_revision  # the revision pinned, or None (read-only)

Anything else (private methods, internal caching, batching strategy) is
implementation detail and may change in 2.7.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CACHE = Path.home() / ".recall" / "models"


class Embedder:
    """Thin wrapper around sentence-transformers. Minimal in 2.5; rewritten
    behavior-preservingly in 2.7."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model_revision: str | None = None,
        cache_folder: Path | None = None,
    ) -> None:
        # Lazy import: sentence_transformers pulls torch (~25s + ~150 MB),
        # which we don't want triggered just by importing `recall.embed`
        # (e.g. during pytest collection of an unrelated test file).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model_revision = model_revision
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

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an (N, dim) float32 array, L2-normalized.

        Normalization on output means cosine similarity reduces to dot
        product downstream — sqlite-vec uses L2 distance by default but
        with normalized vectors L2 and cosine rank identically, so the
        same vectors work for both.
        """
        out: np.ndarray = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # sentence-transformers may return float64 on some platforms; coerce
        # to float32 to match sqlite-vec's FLOAT[N] storage.
        return np.asarray(out, dtype=np.float32)


__all__ = ("DEFAULT_MODEL", "Embedder")
