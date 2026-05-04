"""``Ranker`` protocol and the shared tokenizer used by lexical rankers.

Phase 2 of Recall ships five rankers — semantic (the value prop) plus
four lexical baselines. All implement ``Ranker``; the eval harness
loops generic over a ranker. Reporting all five against the same
corpus and queries is what makes the value-prop delta legible:
"semantic vs the strongest lexical baseline" is the headline number,
"semantic vs four lexical baselines on a ladder" is the receipts.

The ``fts5_unicode61_tokenize`` function is the **shared** tokenizer
between the ``token-overlap`` and ``bm25`` rankers — both must see the
same tokens for the two numbers to be directly comparable on the same
corpus. It's a Python-side approximation of SQLite FTS5's
``tokenize='unicode61 remove_diacritics 1'``; the tiny edge-case
mismatch is documented in CLAUDE.md "Eval harness".
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

# Alphanumeric Unicode letter/digit sequences (excluding underscore, like
# FTS5 unicode61). Diacritics are removed via NFKD normalization before
# this regex runs, and the input is casefolded — so this matches the
# tokens FTS5's unicode61 produces, modulo a handful of edge cases (a
# few Unicode code points whose Letter/Digit classification differs
# slightly between Python's regex engine and SQLite's table).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fts5_unicode61_tokenize(text: str) -> list[str]:
    """Lower-case, diacritic-stripped, alphanumeric tokens.

    Mirror-image of SQLite FTS5's ``tokenize='unicode61 remove_diacritics 1'``
    so token-overlap and BM25 rankers see the same tokens. Used by both;
    do not let them drift.
    """
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return _TOKEN_RE.findall(stripped.casefold())


@runtime_checkable
class Ranker(Protocol):
    """A retrieval ranker over a corpus of strings.

    Lifecycle: instantiate, then ``index(corpus)`` once, then ``search``
    one or more times. Rankers are stateful (the index lives in the
    ranker instance) and not thread-safe.
    """

    name: str

    def index(self, corpus: Sequence[str]) -> None:
        """Build the in-memory index from ``corpus``. Must be called once
        before any ``search``. ``corpus[i]`` is the document with index ``i``."""

    def search(self, queries: Sequence[str], k: int) -> Sequence[Sequence[int]]:
        """For each query, return up to ``k`` corpus indices in rank order
        (best first). Empty list (not None) when no matches."""


__all__ = ("Ranker", "fts5_unicode61_tokenize")
