"""Naive substring baseline — the strawman of the four-lexical ladder.

Splits the NL query on whitespace (no tokenization beyond that — that's
the "naive" part), lowercases, then ranks each corpus item by how many
query words appear as substrings of it. Almost certainly the weakest
baseline; reported for ladder completeness so a skeptical reader can
see the full ramp from "naive" to "fuzzy" to "BM25".

Implementation parallelizes the search loop across processes via
``multiprocessing.Pool``. Single-threaded the ranker is ~85s on
M-series Mac for nl2bash — most of the all-rankers wall-clock budget.
Multi-threaded with all cores (~8 perf cores on M1 Pro / similar) it
drops to ~12-15s, leaving room in the budget for the other rankers.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from multiprocessing import Pool


def _score_chunk(
    args: tuple[Sequence[str], Sequence[str], int],
) -> list[list[int]]:
    """Score one chunk of queries against the full lowercased corpus.

    Module-level so it's picklable for ``multiprocessing.Pool``.
    """
    queries, corpus_lower, k = args
    out: list[list[int]] = []
    for q in queries:
        words = [w for w in q.lower().split() if w]
        if not words:
            out.append([])
            continue
        scored: list[tuple[int, int]] = []
        for i, c_lower in enumerate(corpus_lower):
            count = sum(1 for w in words if w in c_lower)
            if count > 0:
                scored.append((i, count))
        scored.sort(key=lambda x: -x[1])
        out.append([i for i, _ in scored[:k]])
    return out


class NaiveSubstringRanker:
    """Rank by count of query whitespace-words appearing as substrings."""

    name = "naive"

    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._corpus_lower: list[str] = []

    def index(self, corpus: Sequence[str]) -> None:
        self._corpus = list(corpus)
        self._corpus_lower = [c.lower() for c in self._corpus]

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        queries_list = list(queries)
        # Small inputs: skip the multiprocessing overhead.
        if len(queries_list) < 100:
            return _score_chunk((queries_list, self._corpus_lower, k))
        n_workers = max(1, (os.cpu_count() or 1) - 1)
        chunk_size = max(1, len(queries_list) // n_workers + 1)
        chunks = [queries_list[i : i + chunk_size] for i in range(0, len(queries_list), chunk_size)]
        with Pool(processes=n_workers) as pool:
            chunk_results = pool.map(
                _score_chunk,
                [(c, self._corpus_lower, k) for c in chunks],
            )
        return [r for chunk_out in chunk_results for r in chunk_out]


__all__ = ("NaiveSubstringRanker",)
