"""``HistorySource`` protocol and the ``Entry`` model.

All source readers (zsh, bash, fish, atuin) implement ``HistorySource``.
The Phase 2 indexer consumes ``Entry`` instances, scrubs ``text``, computes
the dedup hash, and writes to ``commands``.

Design notes (locked, see CLAUDE.md §2a):

- ``Entry.text`` is the RAW, unscrubbed command. Sources never scrub —
  that's the indexer's job. This keeps the dedup-hash input identical
  across reindex passes regardless of which source reported the entry.
- ``Entry.ts`` is wall-clock unix seconds, the only definition that
  survives multi-source merging in the indexer. Sources that store other
  units (atuin's nanoseconds, etc.) convert at the iter_entries boundary.
- ``ts == 0`` means "unknown" and is always yielded regardless of the
  ``since`` filter — the indexer's UNIQUE constraint catches duplicates.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict


class Entry(BaseModel):
    """One shell-history entry as yielded by a HistorySource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    ts: int  # wall-clock unix seconds; 0 if unknown
    source: str  # 'zsh' | 'bash' | 'fish' | 'atuin'
    source_id: str | None = None  # atuin row id; None for histfile sources
    cwd: str | None = None
    hostname: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    session_id: str | None = None


@runtime_checkable
class HistorySource(Protocol):
    """A read-only view over a single shell-history source.

    Sources are stateless across calls — each ``iter_entries`` opens
    whatever file or DB it needs and closes when the iterator is exhausted.
    """

    name: str

    def iter_entries(self, since: int | None = None) -> Iterator[Entry]:
        """Yield entries with ``ts > since`` in source-native order.

        ``since`` is wall-clock unix seconds. Sources MAY emit entries
        with ``ts <= since`` if they detect a cursor mismatch (histfile
        rewrite, etc.) — backfill-on-mismatch is encapsulated per source.
        Entries with unknown timestamps emit ``ts = 0`` and are always
        yielded regardless of ``since``.
        """
        ...


__all__ = ("Entry", "HistorySource")
