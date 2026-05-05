"""Tests for ``SemanticRanker`` that don't require the heavy embed lane.

Uses a deterministic fake Embedder satisfying the same duck-typed
surface as ``recall.embed.Embedder`` (encode → np.ndarray, dim,
model_name, model_revision). This keeps cache-behavior tests in the
fast lane: end-to-end semantic correctness is covered by
``test_eval.py`` and ``test_embed_behavior_preservation.py`` (both
``@pytest.mark.embed``).

The cache tests pin the opt-in semantic of ``query_cache_size``: 0 means
disabled; > 0 means an LRU bounded at that capacity.
"""

from __future__ import annotations

import hashlib

import numpy as np

from recall.retrieve.semantic import SemanticRanker


class _FakeEmbedder:
    """Deterministic fake Embedder for cache tests.

    Maps each text to a unique normalized vector via SHA256. Counts
    encode calls so tests can assert cache-hit avoidance.
    """

    model_name = "fake-model"
    model_revision = None
    dim = 16

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode(self, texts):
        self.encode_calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # Use first 16 bytes as float8 source, normalize.
            arr = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
            arr = arr - 128.0  # center around zero
            arr = arr / (np.linalg.norm(arr) or 1.0)
            out[i] = arr.astype(np.float32)
        return out


def test_default_no_query_cache() -> None:
    """Default constructor: no query cache (matches 2.6 behavior)."""
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake)
    assert ranker._query_cache is None
    assert ranker._query_cache_size == 0


def test_explicit_zero_size_disables_cache() -> None:
    """Passing ``query_cache_size=0`` explicitly is the same as default."""
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=0)
    assert ranker._query_cache is None


def test_explicit_positive_size_enables_cache() -> None:
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=100)
    assert ranker._query_cache is not None
    assert ranker._query_cache_size == 100


def test_cache_hit_avoids_re_encode() -> None:
    """Querying the same text twice across two search() calls hits the cache.

    encode() is called twice in total: once for index(), once for the first
    search()'s misses. The second search() finds the same text in cache and
    does NOT call encode().
    """
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=10)
    ranker.index(["alpha cmd", "beta cmd", "gamma cmd"])
    # index() consumed one encode call.
    assert len(fake.encode_calls) == 1

    ranker.search(["find the alpha"], k=3)
    # search() encoded the one miss.
    assert len(fake.encode_calls) == 2
    assert fake.encode_calls[1] == ["find the alpha"]

    ranker.search(["find the alpha"], k=3)
    # Second search: cache hit, no new encode call.
    assert len(fake.encode_calls) == 2


def test_cache_partial_hit_only_encodes_misses() -> None:
    """Mix of cache hits and misses: only the misses are sent to encode()."""
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=10)
    ranker.index(["a", "b", "c"])
    ranker.search(["query one"], k=3)
    # Cache now holds "query one". Index uses 1 call, first search uses 1.
    assert len(fake.encode_calls) == 2

    # Second search: one hit, two misses.
    ranker.search(["query one", "query two", "query three"], k=3)
    # Should have triggered exactly one more encode() call with the 2 misses.
    assert len(fake.encode_calls) == 3
    assert fake.encode_calls[2] == ["query two", "query three"]


def test_cache_lru_eviction() -> None:
    """Capacity 2: inserting a third entry evicts the least-recently-used."""
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=2)
    ranker.index(["x"])
    ranker.search(["q1", "q2"], k=1)
    assert ranker._query_cache is not None
    assert set(ranker._query_cache.keys()) == {"q1", "q2"}

    # Add a third miss → q1 (oldest) should evict.
    ranker.search(["q3"], k=1)
    assert set(ranker._query_cache.keys()) == {"q2", "q3"}

    # Re-querying q1 is now a miss again.
    n_calls_before = len(fake.encode_calls)
    ranker.search(["q1"], k=1)
    assert len(fake.encode_calls) == n_calls_before + 1


def test_cache_lru_touch_on_hit() -> None:
    """Hitting a cached entry marks it most-recent → it survives next eviction."""
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=2)
    ranker.index(["x"])
    ranker.search(["q1", "q2"], k=1)
    # Now hit q1 (touch it as MRU).
    ranker.search(["q1"], k=1)
    # Insert q3 → q2 (now LRU) should evict, NOT q1.
    ranker.search(["q3"], k=1)
    assert ranker._query_cache is not None
    assert set(ranker._query_cache.keys()) == {"q1", "q3"}


def test_cache_preserves_input_order_on_partial_hit() -> None:
    """search() result-order matches input order even when some queries hit cache.

    Order-preservation is load-bearing — the runner's per-query metrics
    are zipped with the cases by index. Mismatched order = silently wrong
    metrics.
    """
    fake = _FakeEmbedder()
    ranker = SemanticRanker(embedder=fake, query_cache_size=10)
    ranker.index(["alpha", "beta", "gamma", "delta"])
    # Prime cache with q1 only.
    ranker.search(["q1"], k=2)
    # Mixed query order: hit, miss, hit, miss.
    results = ranker.search(["q1", "q2", "q1", "q3"], k=2)
    assert len(results) == 4
    # Cache invariant: identical query → identical result list.
    assert list(results[0]) == list(results[2])
