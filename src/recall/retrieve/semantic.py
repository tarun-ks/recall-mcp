"""Semantic ranker — the value prop. Wraps ``recall.embed.Embedder`` and
uses an in-memory ``sqlite-vec`` index for KNN search.

Behavior must match Commit 2.5's inline semantic search to within
±0.0001 recall@5 across embed.py rewrites — the behavior-preservation
gate that makes "rewrite under frozen API" testable rather than vibes.

Optional query cache (added 2.7) is for the MCP-server hot path:
"same NL asked twice in a session." Default disabled
(``query_cache_size=0``) so eval-time use, where every query is unique,
doesn't pay LRU bookkeeping for zero hit rate.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from collections.abc import Sequence

import numpy as np
import sqlite_vec

from recall.embed import Embedder


class SemanticRanker:
    """``Ranker`` over a sentence-transformers embedder + sqlite-vec index."""

    name = "semantic"

    def __init__(
        self,
        embedder: Embedder | None = None,
        query_cache_size: int = 0,
    ) -> None:
        self._embedder = embedder if embedder is not None else Embedder()
        self.model_name: str = self._embedder.model_name
        self.model_revision: str | None = self._embedder.model_revision
        self._dim: int = self._embedder.dim
        self._conn: sqlite3.Connection | None = None
        # Query cache (opt-in, default disabled). LRU semantics via
        # OrderedDict.move_to_end + popitem(last=False). None when disabled
        # so the search-loop fast path doesn't pay the dict lookup.
        self._query_cache_size: int = query_cache_size
        self._query_cache: OrderedDict[str, np.ndarray] | None = (
            OrderedDict() if query_cache_size > 0 else None
        )

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

    def _encode_with_cache(self, queries: Sequence[str]) -> np.ndarray:
        """Embed ``queries``, hitting the cache for repeated texts.

        Preserves input order on output: the result at row i is the
        embedding for queries[i]. Cache hits skip the embedder; misses
        are batched into a single embedder.encode() call so we still
        get the C-level batch throughput on the unique-text portion.
        """
        cache = self._query_cache
        assert cache is not None  # guarded by caller
        # Partition by cache hit/miss while preserving original order.
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        hits: dict[int, np.ndarray] = {}
        for i, q in enumerate(queries):
            if q in cache:
                cache.move_to_end(q)  # LRU touch
                hits[i] = cache[q]
            else:
                miss_indices.append(i)
                miss_texts.append(q)

        if miss_texts:
            miss_emb = self._embedder.encode(miss_texts)
            for j, (i, q) in enumerate(zip(miss_indices, miss_texts, strict=True)):
                vec = miss_emb[j]
                hits[i] = vec
                cache[q] = vec
                # Evict oldest if over capacity.
                while len(cache) > self._query_cache_size:
                    cache.popitem(last=False)

        # Reassemble in input order.
        out = np.stack([hits[i] for i in range(len(queries))], axis=0)
        return out.astype(np.float32, copy=False)

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        if self._conn is None:
            raise RuntimeError("index() must be called before search()")
        # Batch-encode all queries upfront — one sentence-transformers call
        # for N queries is much cheaper than N calls (per-call setup overhead
        # is significant). Query encoding remains internal to the ranker;
        # the runner sees a single search() call and times it as one stage.
        if self._query_cache is None:
            query_emb = self._embedder.encode(list(queries))
        else:
            query_emb = self._encode_with_cache(queries)
        results: list[list[int]] = []
        for qvec in query_emb:
            rows = self._conn.execute(
                "SELECT id FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (qvec.tobytes(), k),
            ).fetchall()
            results.append([r[0] for r in rows])
        return results


__all__ = ("SemanticRanker",)
