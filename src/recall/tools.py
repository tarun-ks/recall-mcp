"""recall MCP server — tool implementations.

Phase 3 Commit 3.10: six tools per the locked surface in CLAUDE.md
"MCP tool surface (locked signatures)".

    search             — semantic search over the index (+ optional cwd/host/since filters)
    find_in_project    — semantic search scoped to a working-directory subtree
    commands_after     — given a substring pattern, return commands that
                         followed (same-session, time-ordered)
    failed_recently    — recently exited-non-zero commands (atuin only)
    command_stats      — counts/cwds/success-rate/by-source for a substring pattern
    recent             — most-recent commands (optional cwd/host filter)

ARCHITECTURE NOTES (locked Phase-3 §§1-10)

  Q2 schemas:    Pydantic 2 BaseModel with ``model_json_schema()``;
                 strict bounds on every input string and limit. Validation
                 errors surface as ``isError=True`` CallToolResult payloads
                 (clean MCP error per CLAUDE.md §5), not stack traces.
  Q3 retrieval:  Tools call sqlite-vec MATCH directly via SQL on the
                 long-lived read-only DB connection. The eval-lane
                 ``SemanticRanker`` (pure-numpy matmul) is NOT reused
                 here — eval and production diverge below ``Embedder``.
                 See CLAUDE.md "Eval vs production retrieval architectures".
  Q5 errors:     Strategy (a) — server starts even with no/stale index;
                 each tool checks ``state.has_index`` / ``state.stale_model``
                 / source presence and returns a structured MCP error
                 with the exact remediation message before touching the DB.
  Q7 lifecycle:  Lazy embedder load on first semantic call; subsequent
                 calls reuse the cached singleton via ``encode_async``.
                 ``encode_async`` wraps the encode call in
                 ``redirect_stdout(sys.stderr)`` so any library that
                 *does* try to print can't break stdio framing.

ORDERING / TIE-BREAKING

  Result ordering follows ``ORDERING_CONVENTION`` (below). Convention
  mirrors the 2.7.5-hotfix tie-break decision (CLAUDE.md §4a "Tie-breaking
  convention"): primary sort by relevance/time, secondary tie-break by
  ``id ASC`` (lower id first). Lower-index-wins is the consistent rule
  across the codebase; deviating per tool would surprise reviewers.

SQL SAFETY

  All user-controllable values reach SQL via sqlite3 ``?`` parameter
  placeholders. No string interpolation. SQL injection is structurally
  impossible. LIKE-pattern values are escaped (``\\``, ``%``, ``_``)
  with ``ESCAPE '\\\\'`` so user paths with metacharacters don't
  accidentally widen matches.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, Tool
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from recall.scrub import scrub
from recall.server import ServerState

_LOG = logging.getLogger("recall.server.tools")

# === Module constants ===

ORDERING_CONVENTION = """\
Result ordering across tools:
  - search / find_in_project: similarity score DESC, then id ASC.
  - recent / commands_after / failed_recently: ts DESC, then id ASC.
  - command_stats top_cwds: count DESC, then cwd ASC.
Secondary id-ASC tie-break mirrors CLAUDE.md §4a (2.7.5-hotfix
"lower-index wins" convention) — deterministic and consistent
across the codebase.
"""

# All user inputs are parameterized via sqlite3 ? placeholders. No string
# interpolation; SQL injection is structurally impossible.

# Error message: template (db_path is variable). Function form prevents the
# foot-gun where an unformatted "{db_path}" string ships to a user.
_NO_INDEX_TEMPLATE = (
    "Index not found at {db_path}. Run 'recall index' to build one from your shell history."
)


def no_index_msg(db_path: Path) -> str:
    """Return the user-facing 'no index' error message with ``db_path`` interpolated."""
    return _NO_INDEX_TEMPLATE.format(db_path=db_path)


# Static error messages (no template variables).
STALE_MODEL_MSG = (
    "Index was built with a different embedding model "
    "({indexed!r}) than is currently configured ({configured!r}). "
    "Run 'recall index --rebuild' to re-embed against the configured model."
)


def stale_model_msg(indexed: str | None, configured: str) -> str:
    return STALE_MODEL_MSG.format(indexed=indexed, configured=configured)


NO_ATUIN_SOURCE_MSG = (
    "failed_recently requires the 'atuin' source (only atuin records exit codes). "
    "Install atuin and run 'recall index --source atuin'."
)

CWD_CONTEXT_MISSING_MSG = (
    "find_in_project needs a working-directory context. Pass an explicit 'cwd' "
    "argument or set the MCP_CLIENT_CWD environment variable when launching the "
    "server."
)

INVALID_PATTERN_MSG = (
    "pattern cannot be bare SQL wildcards (e.g. '%', '%%'); provide a literal substring."
)

# === Pydantic input models ===
#
# All BaseModel subclasses use ``extra='forbid'`` — passing an unknown field
# is rejected at validation. Strict by default; the MCP client sends exactly
# the schema we publish.

_INPUT_MODEL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=False)


class _CwdNormalizer:
    """Mixin: trailing-slash normalization for cwd-shaped fields."""

    @staticmethod
    def _strip_trailing_slash(v: str | None) -> str | None:
        if v is None:
            return None
        # Preserve root '/' as-is; otherwise drop a single trailing slash.
        # '/home/u/proj/' → '/home/u/proj'; '/' → '/'.
        return v.rstrip("/") or "/"


class SearchInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=200)
    cwd_prefix: str | None = Field(default=None, max_length=4096)
    host: str | None = Field(default=None, max_length=255)
    since: str | None = Field(default=None, max_length=64)

    @field_validator("cwd_prefix", mode="after")
    @classmethod
    def _strip_cwd(cls, v: str | None) -> str | None:
        return _CwdNormalizer._strip_trailing_slash(v)


class FindInProjectInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=200)
    cwd: str | None = Field(default=None, max_length=4096)

    @field_validator("cwd", mode="after")
    @classmethod
    def _strip_cwd(cls, v: str | None) -> str | None:
        return _CwdNormalizer._strip_trailing_slash(v)


class CommandsAfterInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    pattern: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=200)


class FailedRecentlyInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    window: str = Field(default="24h", max_length=64)
    pattern: str | None = Field(default=None, max_length=1000)
    limit: int = Field(default=20, ge=1, le=200)


class CommandStatsInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    pattern: str = Field(..., min_length=1, max_length=1000)


class RecentInput(BaseModel):
    model_config = _INPUT_MODEL_CONFIG

    limit: int = Field(default=20, ge=1, le=200)
    cwd_prefix: str | None = Field(default=None, max_length=4096)
    host: str | None = Field(default=None, max_length=255)

    @field_validator("cwd_prefix", mode="after")
    @classmethod
    def _strip_cwd(cls, v: str | None) -> str | None:
        return _CwdNormalizer._strip_trailing_slash(v)


# === Pydantic output models ===


class CommandHit(BaseModel):
    """A single command result. Serializable to MCP structured content."""

    id: int
    text: str  # text_scrubbed; raw text never crosses the MCP boundary
    text_hash: str  # 16-hex-char prefix of BLAKE2b(salt ‖ raw); identifier, not lookup key
    source: str
    ts: int
    cwd: str | None = None
    hostname: str | None = None
    exit_code: int | None = None
    duration_ms: int | None = None
    session_id: str | None = None
    score: float | None = None  # cosine similarity for semantic; None for filter-only


class SequenceHit(BaseModel):
    """A pattern match plus the commands that followed it in the same session."""

    pattern_match: CommandHit
    following: list[CommandHit]


class CommandStats(BaseModel):
    top_cwds: list[tuple[str, int]]  # (cwd, count); top 5 by count
    mean_duration_ms: float | None
    success_rate: float | None  # null if no exit codes recorded
    by_source: dict[str, int]
    total: int


# === Helpers ===

_BARE_WILDCARD_RE = re.compile(r"^%+$")
_RELATIVE_WINDOW_RE = re.compile(r"^(\d+)\s*([smhdw])$")


def _check_pattern_safe(p: str) -> None:
    """Reject patterns that are nothing but ``%`` characters.

    A bare-wildcard pattern would compose with our LIKE-wrapping
    (``'%' || pattern || '%'``) into ``LIKE '%%%'`` etc. — matching
    every row. Not a SQL injection (placeholders prevent that), but
    a footgun that would surprise the client.

    Non-bare patterns containing ``%`` or ``_`` are escaped at SQL
    construction time via ``_escape_like``; users get literal substring
    matching even when their pattern contains LIKE metacharacters.
    """
    if _BARE_WILDCARD_RE.match(p.strip()):
        raise ValueError(INVALID_PATTERN_MSG)


def _escape_like(s: str) -> str:
    """Escape ``\\``, ``%``, ``_`` for LIKE with ESCAPE '\\\\'.

    SQL placeholders prevent injection; this prevents user input that
    happens to contain LIKE metacharacters from accidentally widening
    the match (e.g. searching for ``foo_bar`` shouldn't also match
    ``fooXbar``).
    """
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _parse_since(s: str, *, now: datetime | None = None) -> int:
    """Parse ``since`` into a unix-second cutoff. Inclusive lower bound.

    Accepted formats:
      - relative: ``Ns``/``Nm``/``Nh``/``Nd``/``Nw`` (seconds/minutes/hours/days/weeks)
      - absolute: ISO 8601 (e.g. ``2026-05-01``, ``2026-05-01T10:00:00``)

    Raises ``ValueError`` with a user-facing message on failure.
    """
    s = s.strip()
    if not s:
        raise ValueError("'since' cannot be empty")

    m = _RELATIVE_WINDOW_RE.match(s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
        if now is None:
            now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=n * seconds)
        return int(cutoff.timestamp())

    # Try ISO. fromisoformat() accepts 'YYYY-MM-DD' and 'YYYY-MM-DDTHH:MM:SS'.
    # If the input is naive (no tz), assume UTC — same convention as the rest
    # of the codebase (CLAUDE.md §2a: wall-clock unix seconds is the only
    # axis that survives multi-source merging).
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(
            f"could not parse 'since'={s!r}; expected relative (e.g. '7d', '24h', "
            "'30m') or ISO 8601 (e.g. '2026-05-01', '2026-05-01T10:00:00')"
        ) from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp())


def _parse_window(window: str) -> int:
    """Parse the ``failed_recently.window`` arg into seconds.

    Accepts the relative-form units of ``_parse_since`` only — absolute
    timestamps don't make sense as a 'window length'.
    """
    m = _RELATIVE_WINDOW_RE.match(window.strip())
    if not m:
        raise ValueError(
            f"could not parse 'window'={window!r}; expected relative form like '24h', '7d', '30m'"
        )
    n = int(m.group(1))
    unit = m.group(2)
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]
    return n * seconds


def _row_to_command_hit(row: sqlite3.Row, *, score: float | None = None) -> CommandHit:
    """Map a ``commands`` row → CommandHit, defense-in-depth-scrubbing the text.

    Per CLAUDE.md §1: scrubbing happens at index time AND at query response.
    The DB already holds scrubbed text, but we re-scrub before crossing the
    MCP boundary in case a future code path inserts unscrubbed content.
    """
    text_hash_blob = row["text_hash"]
    # text_hash is stored as BLOB (32 bytes). Surface as 16-char hex prefix —
    # enough for identification (2^64 collision resistance) but not the
    # full hash, since it's derived from raw text + salt.
    hex_prefix = text_hash_blob.hex()[:16] if text_hash_blob else ""
    return CommandHit(
        id=int(row["id"]),
        text=scrub(row["text_scrubbed"]),
        text_hash=hex_prefix,
        source=row["source"],
        ts=int(row["ts"]),
        cwd=row["cwd"],
        hostname=row["hostname"],
        exit_code=row["exit_code"] if row["exit_code"] is not None else None,
        duration_ms=row["duration_ms"] if row["duration_ms"] is not None else None,
        session_id=row["session_id"],
        score=score,
    )


def _state_error_response(message: str) -> tuple[dict[str, Any], int]:
    """Build a structured MCP error payload for a state-derived failure.

    Returned dict goes into ``CallToolResult.structuredContent``;
    matching ``TextContent`` is constructed by the dispatcher.
    """
    return ({"error": message, "results": []}, 0)


def _check_state_for_index(state: ServerState) -> tuple[dict[str, Any], int] | None:
    """Common preamble for tools that need the index. Returns an error
    payload to short-circuit, or ``None`` if the index is healthy."""
    if not state.has_index:
        return _state_error_response(no_index_msg(state.db_path))
    if state.stale_model:
        return _state_error_response(
            stale_model_msg(state.indexed_model_name, state.configured_model_name)
        )
    return None


# Encode-callable type — server.py provides the real one (asyncio.Lock-
# guarded ``encode_async``); tests pass a fake.
EncodeFn = Callable[[Sequence[str]], Awaitable[np.ndarray]]


# === Handler: recent ===


async def _handle_recent(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """List most-recent commands, optionally filtered by cwd/host.

    Doesn't need the embedder; just an indexed-ts ORDER BY.
    """
    inp = RecentInput.model_validate(arguments)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None  # has_index → connection opened in main()

    where_parts: list[str] = []
    params: list[Any] = []
    if inp.cwd_prefix is not None:
        # Prefix-boundary match: cwd equals prefix OR cwd starts with 'prefix/'.
        # Anchored on path separator so '/proj' ≠ '/projector' (the
        # adversarial test case).
        escaped = _escape_like(inp.cwd_prefix)
        where_parts.append("(cwd = ? OR cwd LIKE ? ESCAPE '\\')")
        params.extend([inp.cwd_prefix, escaped + "/%"])
    if inp.host is not None:
        where_parts.append("hostname = ?")
        params.append(inp.host)

    where_sql = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""
    sql = (
        "SELECT id, source, text_scrubbed, text_hash, cwd, hostname, "
        "exit_code, duration_ms, session_id, ts "
        f"FROM commands{where_sql} "
        "ORDER BY ts DESC, id ASC LIMIT ?"
    )
    params.append(inp.limit)
    rows = db_conn.execute(sql, params).fetchall()
    hits = [_row_to_command_hit(r) for r in rows]
    return ({"results": [h.model_dump() for h in hits]}, len(hits))


# === Handler: command_stats ===


async def _handle_command_stats(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """Aggregate stats for a substring pattern."""
    inp = CommandStatsInput.model_validate(arguments)
    _check_pattern_safe(inp.pattern)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None

    pat = "%" + _escape_like(inp.pattern) + "%"

    # Use a single CTE to compute every aggregate in one pass — avoids
    # multiple table scans of the matching subset.
    sql = """
    WITH matches AS (
        SELECT id, source, cwd, exit_code, duration_ms
        FROM commands
        WHERE text_scrubbed LIKE ? ESCAPE '\\'
    )
    SELECT
        (SELECT COUNT(*) FROM matches) AS total,
        (SELECT AVG(duration_ms) FROM matches WHERE duration_ms IS NOT NULL) AS mean_dur,
        (SELECT COUNT(*) FROM matches WHERE exit_code = 0) AS n_success,
        (SELECT COUNT(*) FROM matches WHERE exit_code IS NOT NULL) AS n_with_exit
    """
    summary = db_conn.execute(sql, (pat,)).fetchone()
    total = int(summary["total"])

    # Top 5 cwds.
    cwd_rows = db_conn.execute(
        "SELECT cwd, COUNT(*) AS n FROM commands "
        "WHERE text_scrubbed LIKE ? ESCAPE '\\' AND cwd IS NOT NULL "
        "GROUP BY cwd ORDER BY n DESC, cwd ASC LIMIT 5",
        (pat,),
    ).fetchall()

    # By-source counts.
    src_rows = db_conn.execute(
        "SELECT source, COUNT(*) AS n FROM commands "
        "WHERE text_scrubbed LIKE ? ESCAPE '\\' "
        "GROUP BY source ORDER BY source",
        (pat,),
    ).fetchall()

    success_rate = (
        (int(summary["n_success"]) / int(summary["n_with_exit"]))
        if summary["n_with_exit"]
        else None
    )

    stats = CommandStats(
        top_cwds=[(r["cwd"], int(r["n"])) for r in cwd_rows],
        mean_duration_ms=(float(summary["mean_dur"]) if summary["mean_dur"] is not None else None),
        success_rate=success_rate,
        by_source={r["source"]: int(r["n"]) for r in src_rows},
        total=total,
    )
    return (stats.model_dump(), total)


# === Handler: failed_recently ===


async def _handle_failed_recently(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """Recently exited-non-zero commands. Atuin only (others lack exit codes)."""
    inp = FailedRecentlyInput.model_validate(arguments)
    if inp.pattern is not None:
        _check_pattern_safe(inp.pattern)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None

    # Atuin source presence check — failed_recently requires exit codes.
    has_atuin_row = db_conn.execute(
        "SELECT 1 FROM commands WHERE source = 'atuin' LIMIT 1"
    ).fetchone()
    if not has_atuin_row:
        return _state_error_response(NO_ATUIN_SOURCE_MSG)

    cutoff = int(datetime.now(UTC).timestamp()) - _parse_window(inp.window)

    where_parts = ["source = 'atuin'", "exit_code IS NOT NULL", "exit_code != 0", "ts >= ?"]
    params: list[Any] = [cutoff]
    if inp.pattern is not None:
        where_parts.append("text_scrubbed LIKE ? ESCAPE '\\'")
        params.append("%" + _escape_like(inp.pattern) + "%")

    where_sql = " AND ".join(where_parts)
    sql = (
        "SELECT id, source, text_scrubbed, text_hash, cwd, hostname, "
        "exit_code, duration_ms, session_id, ts "
        f"FROM commands WHERE {where_sql} "
        "ORDER BY ts DESC, id ASC LIMIT ?"
    )
    params.append(inp.limit)
    rows = db_conn.execute(sql, params).fetchall()
    hits = [_row_to_command_hit(r) for r in rows]
    return ({"results": [h.model_dump() for h in hits]}, len(hits))


# === Handler: commands_after ===


async def _handle_commands_after(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """Find commands matching ``pattern``; for each, return the next 3
    commands in the same session, time-ordered. Useful for 'what did I
    run after the failing migration?' queries.

    No regex flag (locked Q2). Pattern is treated as a literal substring.
    """
    inp = CommandsAfterInput.model_validate(arguments)
    _check_pattern_safe(inp.pattern)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None

    pat = "%" + _escape_like(inp.pattern) + "%"
    matches = db_conn.execute(
        "SELECT id, source, text_scrubbed, text_hash, cwd, hostname, "
        "exit_code, duration_ms, session_id, ts "
        "FROM commands WHERE text_scrubbed LIKE ? ESCAPE '\\' "
        "ORDER BY ts DESC, id ASC LIMIT ?",
        (pat, inp.limit),
    ).fetchall()

    sequences: list[SequenceHit] = []
    for m in matches:
        if m["session_id"] is None:
            # Without a session_id we can't define 'after' — return the match
            # alone with empty following list.
            sequences.append(SequenceHit(pattern_match=_row_to_command_hit(m), following=[]))
            continue
        # Next 3 commands in the same session, after the match's ts.
        # Tie-break id ASC keeps ordering deterministic when ts ties (sub-
        # second history entries are common on fast scripts).
        following_rows = db_conn.execute(
            "SELECT id, source, text_scrubbed, text_hash, cwd, hostname, "
            "exit_code, duration_ms, session_id, ts "
            "FROM commands WHERE session_id = ? AND (ts > ? OR (ts = ? AND id > ?)) "
            "ORDER BY ts ASC, id ASC LIMIT 3",
            (m["session_id"], m["ts"], m["ts"], m["id"]),
        ).fetchall()
        sequences.append(
            SequenceHit(
                pattern_match=_row_to_command_hit(m),
                following=[_row_to_command_hit(r) for r in following_rows],
            )
        )

    return (
        {"results": [s.model_dump() for s in sequences]},
        len(sequences),
    )


# === Handler: search ===


_DEFAULT_KNN_OVERFETCH = 5  # multiplier on `limit` for KNN candidates pre-filter


async def _semantic_topk(
    db_conn: sqlite3.Connection,
    query_vec: np.ndarray,
    *,
    k: int,
    where_sql: str,
    where_params: Sequence[Any],
) -> list[tuple[int, float]]:
    """Run a sqlite-vec MATCH and return (command_id, distance) pairs.

    sqlite-vec returns L2 distance with normalized vectors; cosine
    similarity = 1 - (distance^2 / 2). Lower distance = closer match.
    Caller handles the score conversion.

    The post-filter pattern: pull more candidates than ``k`` from
    sqlite-vec, then INNER JOIN ``commands`` with WHERE on the metadata
    filters. With over-fetch=5x, even tight filters typically still
    return ``k`` post-filter results.

    NOTE: sqlite-vec's `embedding MATCH ?` requires a literal `k = N`
    in the same WHERE clause. The match-clause is parameterized via
    placeholders for the vector and k.
    """
    overfetch = k * _DEFAULT_KNN_OVERFETCH
    vec_blob = np.ascontiguousarray(query_vec.astype(np.float32)).tobytes()
    sql = f"""
    SELECT c.id, v.distance
    FROM commands_vec v
    INNER JOIN commands c ON c.id = v.command_id
    WHERE v.embedding MATCH ? AND k = ?
      {where_sql}
    ORDER BY v.distance ASC, c.id ASC
    LIMIT ?
    """
    rows = db_conn.execute(
        sql,
        [vec_blob, overfetch, *where_params, k],
    ).fetchall()
    return [(int(r["id"]), float(r["distance"])) for r in rows]


def _build_search_filters(
    *,
    cwd_prefix: str | None,
    host: str | None,
    since: int | None,
) -> tuple[str, list[Any]]:
    """Build the AND-prefixed WHERE fragment + params for search filters.

    Empty fragment if no filters are set. Emits 'AND ...' so it composes
    with the sqlite-vec MATCH WHERE clause cleanly.
    """
    parts: list[str] = []
    params: list[Any] = []
    if cwd_prefix is not None:
        parts.append("(c.cwd = ? OR c.cwd LIKE ? ESCAPE '\\')")
        params.extend([cwd_prefix, _escape_like(cwd_prefix) + "/%"])
    if host is not None:
        parts.append("c.hostname = ?")
        params.append(host)
    if since is not None:
        parts.append("c.ts >= ?")
        params.append(since)
    if not parts:
        return ("", [])
    return (" AND " + " AND ".join(parts), params)


async def _handle_search(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """Semantic search over the index with optional cwd/host/since filters."""
    inp = SearchInput.model_validate(arguments)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None

    since_ts: int | None = None
    if inp.since is not None:
        try:
            since_ts = _parse_since(inp.since)
        except ValueError as e:
            return _state_error_response(str(e))

    # Embed the query (defense-in-depth scrub the query text before logging
    # but NOT before encoding — we want the raw query embedded).
    query_vec_2d = await encode([inp.query])
    query_vec = query_vec_2d[0]

    where_sql, where_params = _build_search_filters(
        cwd_prefix=inp.cwd_prefix,
        host=inp.host,
        since=since_ts,
    )

    candidates = await _semantic_topk(
        db_conn,
        query_vec,
        k=inp.limit,
        where_sql=where_sql,
        where_params=where_params,
    )

    if not candidates:
        return ({"results": []}, 0)

    # Pull row data for the candidates, preserving the candidate order.
    ids = [cid for cid, _ in candidates]
    placeholders = ",".join("?" * len(ids))
    row_map = {
        int(r["id"]): r
        for r in db_conn.execute(
            f"SELECT id, source, text_scrubbed, text_hash, cwd, hostname, "
            f"exit_code, duration_ms, session_id, ts "
            f"FROM commands WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
    }

    hits: list[CommandHit] = []
    for cid, dist in candidates:
        row = row_map.get(cid)
        if row is None:
            continue  # ID present in commands_vec but not in commands — skip silently
        # cosine_sim = 1 - (L2_dist^2 / 2) for L2-normalized vectors
        cos_sim = max(-1.0, min(1.0, 1.0 - (dist * dist) / 2.0))
        hits.append(_row_to_command_hit(row, score=cos_sim))

    return ({"results": [h.model_dump() for h in hits]}, len(hits))


# === Handler: find_in_project ===


def _resolve_find_in_project_cwd(explicit: str | None) -> str | None:
    """Resolve cwd: explicit arg > MCP_CLIENT_CWD env > server-startup cwd.

    Per CLAUDE.md "MCP tool surface". Returns None if no source resolves
    (caller must surface CWD_CONTEXT_MISSING_MSG).
    """
    if explicit is not None:
        return explicit
    import os

    env = os.environ.get("MCP_CLIENT_CWD")
    if env:
        return _CwdNormalizer._strip_trailing_slash(env)
    # Server-startup cwd: only use if it's a real project dir, not '/'.
    # We use os.getcwd() directly (the server's process cwd at the time
    # this is called, which is whatever the launcher set).
    cwd = os.getcwd()
    if cwd and cwd != "/":
        return _CwdNormalizer._strip_trailing_slash(cwd)
    return None


async def _handle_find_in_project(
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int]:
    """Semantic search scoped to a working-directory subtree."""
    inp = FindInProjectInput.model_validate(arguments)

    err = _check_state_for_index(state)
    if err is not None:
        return err
    assert db_conn is not None

    cwd = _resolve_find_in_project_cwd(inp.cwd)
    if cwd is None:
        return _state_error_response(CWD_CONTEXT_MISSING_MSG)

    # Reuse the search machinery with cwd_prefix = cwd.
    return await _handle_search(
        {"query": inp.query, "limit": inp.limit, "cwd_prefix": cwd},
        state=state,
        db_conn=db_conn,
        encode=encode,
    )


# === Tool registry ===


def _tool_for(name: str, description: str, model: type[BaseModel]) -> Tool:
    return Tool(
        name=name,
        description=description,
        inputSchema=model.model_json_schema(),
    )


TOOLS: list[Tool] = [
    _tool_for(
        "search",
        "Semantic search over indexed shell history. Returns the top "
        "`limit` commands ranked by cosine similarity to the query, "
        "optionally filtered by cwd prefix, hostname, or recency.",
        SearchInput,
    ),
    _tool_for(
        "find_in_project",
        "Semantic search scoped to a project working-directory subtree. "
        "If `cwd` is omitted, falls back to the MCP_CLIENT_CWD env var, "
        "then to the server-startup cwd.",
        FindInProjectInput,
    ),
    _tool_for(
        "commands_after",
        "Find commands matching the substring `pattern`; for each match, "
        "return the next ~3 commands run in the same session. Useful for "
        "'what did I run after the failing migration?' queries.",
        CommandsAfterInput,
    ),
    _tool_for(
        "failed_recently",
        "Recently exited-non-zero commands within `window` (e.g. '24h', '7d'). "
        "Requires the atuin source — only atuin records exit codes.",
        FailedRecentlyInput,
    ),
    _tool_for(
        "command_stats",
        "Aggregate stats (top cwds, mean duration, success rate, by-source "
        "counts) for commands matching the substring `pattern`.",
        CommandStatsInput,
    ),
    _tool_for(
        "recent",
        "Most-recent commands, optionally filtered by cwd prefix or hostname. "
        "Time-ordered DESC, then id ASC for tie-break.",
        RecentInput,
    ),
]


# Handler dispatch — name → coroutine. Single source of truth for the wiring.
HANDLERS: dict[str, Callable[..., Awaitable[tuple[dict[str, Any], int]]]] = {
    "search": _handle_search,
    "find_in_project": _handle_find_in_project,
    "commands_after": _handle_commands_after,
    "failed_recently": _handle_failed_recently,
    "command_stats": _handle_command_stats,
    "recent": _handle_recent,
}


async def dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    state: ServerState,
    db_conn: sqlite3.Connection | None,
    encode: EncodeFn,
) -> tuple[dict[str, Any], int, bool]:
    """Run the named tool handler. Returns (payload, count, is_error).

    - ValidationError (Pydantic) → (error-payload, 0, True)
    - State error (no_index, etc.) → handler returns it directly
    - Internal exception → raised as McpError(INTERNAL_ERROR)

    Logs structured (tool, count, error) per CLAUDE.md §1 trust feature
    (query text NOT logged by default — Q8 logging policy).
    """
    handler = HANDLERS.get(name)
    if handler is None:
        # Should be unreachable: list_tools advertises only the keys we
        # define here; the SDK validates name before dispatch. But guard
        # against a future refactor that drifts the two lists.
        raise McpError(ErrorData(code=INVALID_PARAMS, message=f"unknown tool: {name!r}"))

    try:
        payload, count = await handler(
            arguments,
            state=state,
            db_conn=db_conn,
            encode=encode,
        )
    except ValidationError as e:
        # Surface Pydantic validation errors as a clean MCP error payload,
        # not a stack trace. Per CLAUDE.md §5: 'Bad input returns a clean
        # MCP error, not a stack trace.'
        msg = _format_validation_error(e)
        _LOG.info("tool=%s validation_error=%s", name, msg)
        return ({"error": msg, "results": []}, 0, True)
    except ValueError as e:
        # Pattern/since/window parse failures bubble up as ValueError from
        # the handlers. Same surface as ValidationError — user-facing message.
        _LOG.info("tool=%s value_error=%s", name, e)
        return ({"error": str(e), "results": []}, 0, True)
    except McpError:
        raise
    except Exception as e:
        # Defensive: unexpected internal failure. Log with traceback for
        # post-mortem, surface as INTERNAL_ERROR (-32603).
        _LOG.exception("tool=%s internal_error", name)
        raise McpError(
            ErrorData(code=INTERNAL_ERROR, message=f"internal error in {name}: {e}")
        ) from e

    is_error = "error" in payload
    if is_error:
        _LOG.info("tool=%s state_error count=0 message=%r", name, payload["error"])
    else:
        _LOG.info("tool=%s ok count=%d", name, count)
    return (payload, count, is_error)


def _format_validation_error(e: ValidationError) -> str:
    """Compress Pydantic's verbose errors into a one-liner per field."""
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts) or "invalid input"


__all__ = (
    "HANDLERS",
    "ORDERING_CONVENTION",
    "TOOLS",
    "CommandHit",
    "CommandStats",
    "CommandStatsInput",
    "CommandsAfterInput",
    "FailedRecentlyInput",
    "FindInProjectInput",
    "RecentInput",
    "SearchInput",
    "SequenceHit",
    "dispatch_tool",
    "no_index_msg",
    "stale_model_msg",
)


# Quiet linter on the asyncio import: it's used by the type alias resolution
# of EncodeFn (Awaitable). Keeping the import explicit guards against future
# refactors that lose track.
_ = asyncio
