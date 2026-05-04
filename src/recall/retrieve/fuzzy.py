"""Fuzzy substring baseline — rapidfuzz's ``partial_ratio`` over the
whole corpus per query.

Approximates fzf's substring-fuzzy behavior. **Not** an exact match for
fzf's scoring algorithm — fzf adds bonuses for word boundaries and
camelCase splits that ``partial_ratio`` doesn't. The user-facing claim
is "we beat what zsh+fzf users do today" framed as "fzf-like, not
exactly fzf"; if we ever need higher fidelity we can shell out to the
fzf binary or evaluate ``pyfzf``. See CLAUDE.md "Eval harness".

Why ``partial_ratio`` specifically: it captures the "is the query a
fuzzy substring of the command" intent. The closely-named ``WRatio``
combines tokenization and partial matching, which makes it stronger
but moves it away from fzf's pure-subsequence model.

Implementation uses ``rapidfuzz.process.cdist`` rather than per-query
``process.extract``. ``cdist`` does the whole pairwise scoring matrix
in one C kernel; ``extract`` per query was O(n_corpus) Python
overhead × n_queries = catastrophic on nl2bash (~5+ minutes vs ~10s).
"""

from __future__ import annotations

from collections.abc import Sequence


class FuzzyRanker:
    """Rank by rapidfuzz ``partial_ratio`` (batched via ``process.cdist``)."""

    name = "fuzzy"

    def __init__(self) -> None:
        self._corpus: list[str] = []

    def index(self, corpus: Sequence[str]) -> None:
        self._corpus = list(corpus)

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        # Lazy import: rapidfuzz import takes ~1s; importing it at module-
        # load time would balloon scrub-canary collection. Same lesson as
        # the sentence-transformers lazy-import in 2.5 (CLAUDE.md
        # "Composition is where bugs live", 2.5 named instance).
        import numpy as np
        from rapidfuzz import fuzz, process

        # Batch matrix: shape (n_queries, n_corpus), float64 scores 0-100.
        # One C kernel does all pairwise comparisons; workers=-1 parallelizes
        # across all cores (necessary on M-series — single-threaded
        # partial_ratio over 11k × 10k pairs is ~5+ minutes; multi-threaded
        # is ~30-60s).
        matrix = process.cdist(
            list(queries),
            self._corpus,
            scorer=fuzz.partial_ratio,
            workers=-1,
        )
        results: list[list[int]] = []
        n_corpus = matrix.shape[1]
        for q_idx in range(matrix.shape[0]):
            scores = matrix[q_idx]
            # argpartition gets top-k unsorted in O(n); then sort just those k.
            if k >= n_corpus:
                top_idx = np.argsort(-scores, kind="stable")
            else:
                # Indices of the k highest scores (unordered among themselves).
                top_unsorted = np.argpartition(-scores, k)[:k]
                # Order those k by descending score (stable for ties).
                top_idx = top_unsorted[np.argsort(-scores[top_unsorted], kind="stable")]
            # Drop zero-score entries — no match at all.
            results.append([int(i) for i in top_idx if scores[i] > 0])
        return results


__all__ = ("FuzzyRanker",)
