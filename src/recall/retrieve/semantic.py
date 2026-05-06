"""Semantic ranker — the value prop. Pure numpy in-memory KNN over the
embedder's normalized output.

Behavior must match Commit 2.5's inline semantic search to within
±0.0001 recall@5 — the behavior-preservation gate that makes "rewrite
under frozen Ranker protocol" testable rather than vibes.

ARCHITECTURAL DIVERGENCE (2.7.5): EVAL VS PRODUCTION INDEXER

    Eval path (this file): source → embedder → in-memory numpy
    matmul → top-k via argpartition. No sqlite-vec; no DB. The corpus
    is a numpy array, the search is one BLAS call.

    Production indexer path (Commit 2.8, future): persistent on-disk
    sqlite-vec virtual table for KNN over 50k+ commands across recall
    invocations.

    The two paths share an embedder but diverge in their KNN data
    structure. Eval-time use cases (small N, in-memory, repeated runs)
    favor pure-numpy simplicity; production use cases (large N,
    persistent storage, single-query latency) need sqlite-vec's
    on-disk index.

EQUIVALENCE GUARANTEE (per CLAUDE.md §4a)

    Top-5 IDs from this implementation are equal as a SET (not as an
    ordered list) to the sqlite-vec MATCH reference across all 11,348
    nl2bash queries. List equality is NOT claimed because float32
    cosine similarity permits ties, and sqlite-vec / argpartition
    differ in how ties are broken. Set equality + the recall@5
    behavior-preservation gate together pin the algorithm.

Optional query cache (added 2.7) is for the MCP-server hot path:
"same NL asked twice in a session." Default disabled
(``query_cache_size=0``) so eval-time use, where every query is
unique, doesn't pay LRU bookkeeping for zero hit rate.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import numpy as np

from recall.embed import Embedder


class SemanticRanker:
    """``Ranker`` over a sentence-transformers embedder + numpy KNN."""

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
        self._corpus_emb: np.ndarray | None = None
        # Query cache (opt-in, default disabled). LRU semantics via
        # OrderedDict.move_to_end + popitem(last=False). None when disabled
        # so the search-loop fast path doesn't pay the dict lookup.
        self._query_cache_size: int = query_cache_size
        self._query_cache: OrderedDict[str, np.ndarray] | None = (
            OrderedDict() if query_cache_size > 0 else None
        )

    def index(self, corpus: Sequence[str]) -> None:
        # Encode + cache the corpus matrix in memory. Embedder normalizes
        # to unit length, so cosine similarity reduces to dot product.
        self._corpus_emb = self._embedder.encode(list(corpus))

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
        if self._corpus_emb is None:
            raise RuntimeError("index() must be called before search()")
        if not queries:
            return []

        # Batch-encode queries (cache opt-in; eval path is all-misses).
        if self._query_cache is None:
            query_emb = self._embedder.encode(list(queries))
        else:
            query_emb = self._encode_with_cache(queries)
        # query_emb: (n_queries, dim) float32

        # Single batched matmul: full (n_corpus, n_queries) similarity matrix.
        # Memory: n_corpus × n_queries × 4 bytes. nl2bash peak ~482 MB; CI
        # ubuntu-latest has 7 GB RAM. Chunking deferred until any single
        # eval workload approaches 1 GB peak (CLAUDE.md deferred items).
        sims = self._corpus_emb @ query_emb.T  # (n_corpus, n_queries) float32

        # Defensive guard: bge-small outputs unit-normalized vectors so dot
        # products are bounded [-1, 1] and finite. Asserting catches any
        # embedder pathology (rare; cheap insurance).
        assert np.isfinite(sims).all(), "embedder produced non-finite similarities"

        # Top-k via argpartition + stable secondary argsort.
        # kind="stable" ensures index-based tie-breaking; required for
        # cross-platform reproducibility per the equivalence-test
        # contract (CLAUDE.md §4a — set equality, not list equality).
        n_corpus = sims.shape[0]
        if k >= n_corpus:
            order = np.argsort(-sims, axis=0, kind="stable")
            top = order[:n_corpus]
        else:
            partial = np.argpartition(-sims, k, axis=0)[:k, :]
            top_scores = np.take_along_axis(sims, partial, axis=0)
            order = np.argsort(-top_scores, axis=0, kind="stable")
            top = np.take_along_axis(partial, order, axis=0)
        # top: (k, n_queries) int64

        # Per-query result lists in input order, top-down.
        return [top[:, q].tolist() for q in range(top.shape[1])]


__all__ = ("SemanticRanker",)
