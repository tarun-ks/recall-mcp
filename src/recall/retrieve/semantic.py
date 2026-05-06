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

# Score-precision rounding (2.7.5 hotfix). 5 decimals = 1e-5 precision;
# 1000× finer than meaningful score differences, 100× coarser than BLAS
# epsilon variance. See ``search()`` for the full rationale.
_SCORE_ROUND_DECIMALS = 5
# Index-tiebreaker scale. Must be << 10^-_SCORE_ROUND_DECIMALS so score
# dominates, and >> float64 epsilon (~1e-16) so the perturbation isn't
# rounded away. 1e-10 satisfies both for n_corpus up to ~10^5.
_INDEX_SCALE = 1e-10


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

        # Top-k via deterministic composite-key argsort.
        #
        # CROSS-RUNNER DETERMINISM (CLAUDE.md "Composition is where bugs live"
        # instance #8). The 2.7.5 verify-branch CI passed with delta = -1e-04;
        # the post-merge main CI run on identical content landed at -2.6e-04
        # — outside the ±0.0001 behavior-preservation tolerance. Root cause:
        # two-stage variance composition.
        #
        #   (1) BLAS matmul rounding order produces tiny per-element variance
        #       (~1e-7 on individual cosine scores) — different CI runners,
        #       different thread interleavings, identical inputs.
        #   (2) np.argpartition's "unspecified order among equal elements"
        #       semantic amplifies that 1e-7 input variance into top-5 set
        #       divergence at near-tied score boundaries — and that
        #       propagates to recall@5 drift past the tolerance gate.
        #
        # Primitives correct in isolation (BLAS matmul, argpartition); the
        # bug lives at the boundary. Fix: compose them differently.
        #
        # TWO-STEP STRUCTURAL FIX:
        #   (a) Round sims to 5-decimal precision (1e-5). This is 1000×
        #       finer than meaningful cosine score differences for
        #       bge-small-en-v1.5 outputs (real ranking distinctions
        #       happen at >= 1e-3 magnitudes), and 100× coarser than BLAS
        #       epsilon variance. Below this granularity, score
        #       differences are noise; above, signal.
        #   (b) Stable argsort over a single composite key encoding
        #       (-rounded_score, index). Index acts as a deterministic
        #       tiebreaker; BLAS variance no longer reaches the ranking.
        #
        # Performance: full O(n log n) argsort instead of O(n)
        # argpartition. nl2bash measured: ~+0.5s on M-series, well within
        # the ≤17s gate. Worth the determinism.
        n_corpus = sims.shape[0]
        sims_rounded = np.round(sims, decimals=_SCORE_ROUND_DECIMALS).astype(np.float32)
        # Composite key per (corpus, query) cell:
        #   primary:   -rounded_score   (ascending sort = best first)
        #   secondary: index * INDEX_SCALE  (smaller index breaks ties)
        # INDEX_SCALE * (n_corpus - 1) << 10^-_SCORE_ROUND_DECIMALS so
        # score is the dominant key and index only matters on rounded
        # ties. Float64 has ~1e-16 epsilon; INDEX_SCALE = 1e-10 is well
        # above float64 noise and well below score precision.
        indices_col = np.arange(n_corpus, dtype=np.float64).reshape(-1, 1)
        composite = -sims_rounded.astype(np.float64) + indices_col * _INDEX_SCALE
        order = np.argsort(composite, axis=0, kind="stable")
        top = order[:k, :] if k < n_corpus else order
        # top: (k, n_queries) int64

        # Per-query result lists in input order, top-down.
        return [top[:, q].tolist() for q in range(top.shape[1])]


__all__ = ("SemanticRanker",)
