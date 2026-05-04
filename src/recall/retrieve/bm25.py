"""BM25 baseline — the academically defensible IR baseline.

Uses SQLite FTS5's built-in ``unicode61 remove_diacritics 1`` tokenizer
and the ``bm25()`` ranking function. The tokenizer is the same one
``recall.retrieve.base.fts5_unicode61_tokenize`` approximates Python-
side for the token-overlap ranker; using FTS5 directly here means
BM25's tokens are exactly what FTS5 says they are (the Python-side
function is the approximation, not the source of truth).

Query tokens are joined with ``OR`` rather than the FTS5 default ``AND``
because nl2bash is paraphrastic — requiring every query token to appear
in the matched command would push recall to near-zero on most queries.
``OR`` semantics give BM25 its proper "rank by sum of per-token scores"
behavior, which is the IR-standard interpretation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

from recall.retrieve.base import fts5_unicode61_tokenize


class Bm25Ranker:
    """Rank by FTS5's ``bm25()`` over a unicode61-tokenized corpus."""

    name = "bm25"

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None

    def index(self, corpus: Sequence[str]) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE VIRTUAL TABLE bm25_idx USING fts5("
            "text, tokenize='unicode61 remove_diacritics 1')"
        )
        # 1-indexed rowid so we can subtract 1 at search time to get
        # a 0-indexed corpus position. SQLite tolerates rowid=0 but FTS5
        # has historical quirks; using positive rowids is safer.
        conn.executemany(
            "INSERT INTO bm25_idx(rowid, text) VALUES (?, ?)",
            [(i + 1, c) for i, c in enumerate(corpus)],
        )
        self._conn = conn

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        if self._conn is None:
            raise RuntimeError("index() must be called before search()")
        results: list[list[int]] = []
        for q in queries:
            tokens = fts5_unicode61_tokenize(q)
            if not tokens:
                results.append([])
                continue
            # Quote each token to neutralize any FTS5 syntax characters
            # that survived tokenization. Then OR them together — see
            # module docstring for why OR rather than AND.
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            try:
                rows = self._conn.execute(
                    "SELECT rowid FROM bm25_idx WHERE bm25_idx MATCH ? "
                    "ORDER BY bm25(bm25_idx) LIMIT ?",
                    (fts_query, k),
                ).fetchall()
            except sqlite3.OperationalError:
                # FTS5 syntax error after tokenization → no matches for
                # this query (rare, but possible with pathological input).
                results.append([])
                continue
            results.append([r[0] - 1 for r in rows])
        return results


__all__ = ("Bm25Ranker",)
