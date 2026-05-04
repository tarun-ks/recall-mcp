"""Semantic ranker — the value prop. Wraps ``recall.embed.Embedder`` and
uses an in-memory ``sqlite-vec`` index for KNN search.

Behavior at this commit (2.6) MUST match the inline semantic search
that Commit 2.5's eval harness performed — the runner refactor is
behavior-preserving and the behavior-preservation gate is "semantic
recall@5 matches 2.5's value to ±0.0001."
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import sqlite_vec

from recall.embed import Embedder


class SemanticRanker:
    """``Ranker`` over a sentence-transformers embedder + sqlite-vec index."""

    name = "semantic"

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder if embedder is not None else Embedder()
        self.model_name: str = self._embedder.model_name
        self.model_revision: str | None = self._embedder.model_revision
        self._dim: int = self._embedder.dim
        self._conn: sqlite3.Connection | None = None

    def index(self, corpus: Sequence[str]) -> None:
        corpus_emb = self._embedder.encode(list(corpus))
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"CREATE VIRTUAL TABLE vec USING vec0("
            f"id INTEGER PRIMARY KEY, embedding FLOAT[{self._dim}])"
        )
        conn.executemany(
            "INSERT INTO vec(id, embedding) VALUES (?, ?)",
            [(i, vec.tobytes()) for i, vec in enumerate(corpus_emb)],
        )
        self._conn = conn

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        if self._conn is None:
            raise RuntimeError("index() must be called before search()")
        # Batch-encode all queries upfront — one sentence-transformers call
        # for N queries is much cheaper than N calls (per-call setup overhead
        # is significant). Query encoding remains internal to the ranker;
        # the runner sees a single search() call and times it as one stage.
        query_emb = self._embedder.encode(list(queries))
        results: list[list[int]] = []
        for qvec in query_emb:
            rows = self._conn.execute(
                "SELECT id FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (qvec.tobytes(), k),
            ).fetchall()
            results.append([r[0] for r in rows])
        return results


__all__ = ("SemanticRanker",)
