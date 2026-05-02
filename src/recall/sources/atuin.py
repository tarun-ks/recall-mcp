"""Reader for atuin history (SQLite at ``~/.local/share/atuin/history.db``).

Always opens the user's atuin DB read-only with ``immutable=1`` — Recall
never writes to it, never creates journal sidecars (``-wal``, ``-shm``,
``-journal``). Schema is detected by introspecting columns at runtime,
not by hardcoded order, so atuin minor-version drift doesn't break us;
missing required columns surface as a clear ``AtuinSchemaError``.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from recall.sources.base import Entry

_LOG = logging.getLogger("recall.sources.atuin")

_DEFAULT_PATH = Path.home() / ".local" / "share" / "atuin" / "history.db"

_REQUIRED_COLUMNS = frozenset({"timestamp", "command"})
_OPTIONAL_COLUMNS = frozenset({"id", "duration", "exit", "cwd", "session", "hostname"})

# atuin stores timestamps and durations in nanoseconds.
_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000


class AtuinSchemaError(Exception):
    """Raised when the atuin history table doesn't have the expected schema."""


class AtuinSource:
    """``HistorySource`` over an atuin sqlite database."""

    name = "atuin"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else _DEFAULT_PATH

    def iter_entries(self, since: int | None = None) -> Iterator[Entry]:
        if not self.path.exists():
            return
        # ?mode=ro&immutable=1 — read-only, never create journal sidecars
        # against the user's atuin DB. URI mode is required for these flags.
        uri = f"file:{self.path}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.row_factory = sqlite3.Row
            cols = self._validate_schema(conn)
            yield from self._iter(conn, cols, since)
        finally:
            conn.close()

    @staticmethod
    def _validate_schema(conn: sqlite3.Connection) -> frozenset[str]:
        info = conn.execute("PRAGMA table_info(history)").fetchall()
        cols = frozenset(row["name"] for row in info)
        missing = _REQUIRED_COLUMNS - cols
        if missing:
            raise AtuinSchemaError(
                f"atuin history table is missing required columns: "
                f"{sorted(missing)}. Found: {sorted(cols)}. "
                "Your atuin schema may be older or newer than what Recall "
                "supports."
            )
        return cols

    def _iter(
        self,
        conn: sqlite3.Connection,
        cols: frozenset[str],
        since: int | None,
    ) -> Iterator[Entry]:
        select_cols = sorted({"timestamp", "command"} | (cols & _OPTIONAL_COLUMNS))
        # Column names come from the static _REQUIRED/_OPTIONAL allowlists,
        # not user input — safe to interpolate.
        sql = f"SELECT {', '.join(select_cols)} FROM history"
        params: tuple[int, ...] = ()
        if since is not None:
            sql += " WHERE timestamp > ?"
            params = (since * _NS_PER_S,)
        sql += " ORDER BY timestamp"

        for row in conn.execute(sql, params):
            ts = int(row["timestamp"]) // _NS_PER_S
            duration_ms: int | None = None
            if "duration" in cols and row["duration"] is not None:
                duration_ms = int(row["duration"]) // _NS_PER_MS
            yield Entry(
                text=row["command"],
                ts=ts,
                source=self.name,
                source_id=str(row["id"]) if "id" in cols and row["id"] is not None else None,
                cwd=row["cwd"] if "cwd" in cols else None,
                hostname=row["hostname"] if "hostname" in cols else None,
                exit_code=row["exit"] if "exit" in cols else None,
                duration_ms=duration_ms,
                session_id=row["session"] if "session" in cols else None,
            )


__all__ = ("AtuinSchemaError", "AtuinSource")
