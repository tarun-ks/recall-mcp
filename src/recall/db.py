"""SQLite + sqlite-vec storage layer.

Exposes:
    connect(path)          — open or create the recall DB; loads sqlite-vec.
    migrate(conn)          — apply pending migrations; idempotent.
    schema_version(conn)   — current PRAGMA user_version.
    get_meta / set_meta    — typed accessors for the meta table.
    dedup_salt(conn)       — read the dedup salt (auto-init on first call).
    rotate_dedup_salt(conn)— rotate the salt; primitive only — does NOT
                              clear the commands table. The CLI is
                              responsible for the clear-then-rotate-then-
                              re-index sequence (see CLAUDE.md §4a).
    dedup_hash(salt, raw)  — BLAKE2b(salt ‖ raw), 32-byte digest.

Logging goes to stderr / `~/.recall/logs/recall.log` only — never stdout
(the stdio MCP transport reserves stdout for JSON-RPC frames).
"""

from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import blake2b
from importlib.resources import files
from pathlib import Path

import sqlite_vec

from recall.scrub import SCRUBBER_VERSION

_LOG = logging.getLogger("recall.db")

_CURRENT_SCHEMA_VERSION = 1
_DEDUP_SALT_BYTES = 32  # 256 bits
_DEDUP_HASH_DIGEST_SIZE = 32  # BLAKE2b output bytes; matches BLOB column width


def _default_db_path() -> Path:
    """Resolve the default DB path. ``RECALL_DB_PATH`` env override beats the
    built-in default; both are evaluated lazily at every ``connect()`` call."""
    override = os.environ.get("RECALL_DB_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".recall" / "db.sqlite"


class DBError(Exception):
    """Raised for db-layer setup failures (extension load, missing schema, etc.)."""


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension. Fail with a clear message if SQLite
    was built without LOAD_EXTENSION support (often the case for macOS
    system Python — uvx recall-mcp avoids this by shipping its own Python).
    """
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.NotSupportedError) as e:
        raise DBError(
            "Your Python's sqlite3 module was built without extension "
            "loading support, which Recall requires for sqlite-vec. "
            "Install via `uvx recall-mcp` (uv ships a Python with full "
            "SQLite). On macOS, the system Python typically can't load "
            "extensions."
        ) from e
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the recall DB at ``path`` (default
    ``~/.recall/db.sqlite``). Returns a connection with sqlite-vec loaded
    and PRAGMAs set. Caller is responsible for ``migrate(conn)``.
    """
    db_path = path if path is not None else _default_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the meta value for ``key`` or None if absent."""
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return None if row is None else str(row["value"])


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert-or-update a meta entry."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)\n"
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Explicit transaction wrapper for callers that opened the connection
    in autocommit mode (which ``connect()`` does)."""
    conn.execute("BEGIN")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def migrate(conn: sqlite3.Connection) -> None:
    """Apply pending migrations and seed runtime meta. Idempotent.

    On a fresh DB (user_version=0) this applies ``0001_initial.sql`` and
    bumps user_version to 1. On a DB already at the current version, this
    just ensures runtime meta entries (salt, scrubber_version,
    schema_version) are present — leaving any already-set value untouched.
    """
    current = schema_version(conn)
    if current == 0:
        sql = (files("recall.migrations") / "0001_initial.sql").read_text(encoding="utf-8")
        # executescript() issues an implicit COMMIT before running and runs
        # statements in autocommit mode, so don't wrap it in transaction().
        # SQLite DDL is durable per-statement, which is good enough here.
        conn.executescript(sql)
        conn.execute(f"PRAGMA user_version = {_CURRENT_SCHEMA_VERSION}")
        _LOG.info("recall.db: applied schema v%d", _CURRENT_SCHEMA_VERSION)
    elif current > _CURRENT_SCHEMA_VERSION:
        raise DBError(
            f"Database schema is at v{current} but this build only knows "
            f"v{_CURRENT_SCHEMA_VERSION}. Upgrade `recall-mcp` to a newer "
            "version, or run against a fresh DB."
        )
    _ensure_runtime_meta(conn)


def _ensure_runtime_meta(conn: sqlite3.Connection) -> None:
    """Seed the salt and version markers iff they're not already set.
    Never overwrites — this is what guarantees salt preservation across
    every ``migrate()`` call (including the implicit one inside
    ``--rebuild``)."""
    if get_meta(conn, "dedup_salt") is None:
        salt = secrets.token_bytes(_DEDUP_SALT_BYTES)
        with transaction(conn):
            set_meta(conn, "dedup_salt", salt.hex())
            set_meta(conn, "dedup_salt_version", "1")
        _LOG.info("recall.db: generated initial dedup salt")
    if get_meta(conn, "scrubber_version") is None:
        set_meta(conn, "scrubber_version", SCRUBBER_VERSION)
    if get_meta(conn, "schema_version") is None:
        set_meta(conn, "schema_version", str(_CURRENT_SCHEMA_VERSION))


def dedup_salt(conn: sqlite3.Connection) -> bytes:
    """Return the dedup salt as raw bytes. Assumes ``migrate(conn)`` has run."""
    hexstr = get_meta(conn, "dedup_salt")
    if hexstr is None:
        raise DBError(
            "dedup_salt is missing from meta. Did you forget to call "
            "migrate(conn) first? Or has the meta table been tampered with?"
        )
    return bytes.fromhex(hexstr)


def rotate_dedup_salt(conn: sqlite3.Connection) -> bytes:
    """Generate and store a new dedup salt; bump dedup_salt_version. Returns
    the new salt bytes.

    Mechanism only — does NOT clear the commands table. After this call
    runs, any rows still in ``commands`` carry hashes computed with the
    OLD salt; combined with new rows hashed under the new salt, dedup
    breaks silently. The CLI (Commit 2.7) wraps this in a
    clear→rotate→re-index sequence and rejects ``--new-salt`` without
    ``--rebuild`` for exactly this reason. See CLAUDE.md §4a.
    """
    new = secrets.token_bytes(_DEDUP_SALT_BYTES)
    current_v = int(get_meta(conn, "dedup_salt_version") or "0")
    with transaction(conn):
        set_meta(conn, "dedup_salt", new.hex())
        set_meta(conn, "dedup_salt_version", str(current_v + 1))
    _LOG.warning("recall.db: rotated dedup salt to v%d", current_v + 1)
    return new


def dedup_hash(salt: bytes, raw: str) -> bytes:
    """Compute ``BLAKE2b(salt ‖ raw_utf8)`` with a 32-byte digest.

    Pure function. Same input → same output. Uses BLAKE2b's keyed mode
    NOT — we explicitly concatenate so the salt also acts as namespace
    separation between Recall installs (different salts → different hash
    spaces, even on identical history files).
    """
    h = blake2b(digest_size=_DEDUP_HASH_DIGEST_SIZE)
    h.update(salt)
    h.update(raw.encode("utf-8"))
    return h.digest()


__all__ = (
    "DBError",
    "connect",
    "dedup_hash",
    "dedup_salt",
    "get_meta",
    "migrate",
    "rotate_dedup_salt",
    "schema_version",
    "set_meta",
    "transaction",
)
