"""Token-overlap baseline — proper tokenization, count-based ranking.

Tokenizes both query and corpus with the FTS5-equivalent unicode61
tokenizer (shared with the BM25 ranker so the two are directly
comparable on the same corpus). Ranks each corpus item by the size
of the set intersection of its tokens with the query's tokens.

Stronger than naive substring because tokens are normalized: "files"
matches "file" only after stemming (which we don't do); but "FILE"
matches "file" trivially via casefolding. In between substring and BM25
on the lexical ladder.
"""

from __future__ import annotations

from collections.abc import Sequence

from recall.retrieve.base import fts5_unicode61_tokenize


class TokenOverlapRanker:
    """Rank by intersection-count of FTS5-tokenized query / corpus tokens."""

    name = "token-overlap"

    def __init__(self) -> None:
        self._corpus_tokens: list[set[str]] = []

    def index(self, corpus: Sequence[str]) -> None:
        self._corpus_tokens = [set(fts5_unicode61_tokenize(c)) for c in corpus]

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        results: list[list[int]] = []
        for q in queries:
            q_tokens = set(fts5_unicode61_tokenize(q))
            if not q_tokens:
                results.append([])
                continue
            scored: list[tuple[int, int]] = []
            for i, ct in enumerate(self._corpus_tokens):
                overlap = len(q_tokens & ct)
                if overlap > 0:
                    scored.append((i, overlap))
            scored.sort(key=lambda x: -x[1])
            results.append([i for i, _ in scored[:k]])
        return results


__all__ = ("TokenOverlapRanker",)
