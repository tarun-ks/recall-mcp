"""Tests for the six MCP tool implementations in ``recall.tools``.

These run in-process against a real SQLite DB built per-test by inserting
rows directly into the migrated schema. The Embedder is replaced with a
deterministic stub so the eval-lane model load doesn't bleed into this
test file's collection budget — same discipline as test_indexer.py.

Coverage outline (matches Phase 3 §3.10 plan):
  - Pydantic validation: bounds, wildcard rejection, trailing-slash
    normalization
  - Adversarial prefix-boundary: '/proj' matches '/proj', '/proj/sub'
    NOT '/projector'; '/proj/' (trailing slash) normalizes to same
  - LIKE escape: path with '_' / '%' doesn't accidentally widen
  - State errors: no_index, stale_model, no_atuin_source, cwd_context
  - _parse_since (relative + absolute) and _parse_window
  - Handler bodies with a real fixture DB (recent, command_stats,
    failed_recently, commands_after, search, find_in_project)
  - Defense-in-depth scrubbing on the output path
  - text_hash hex prefix is exactly 16 chars
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError

from recall.db import connect, migrate, set_meta
from recall.embed import DEFAULT_MODEL
from recall.server import ServerState
from recall.tools import (
    CommandsAfterInput,
    CommandStatsInput,
    FailedRecentlyInput,
    FindInProjectInput,
    RecentInput,
    SearchInput,
    _check_pattern_safe,
    _escape_like,
    _parse_since,
    _parse_window,
    _resolve_find_in_project_cwd,
    dispatch_tool,
    no_index_msg,
    stale_model_msg,
)

# === Test fixtures ===


def _make_state(
    *,
    db_path: Path,
    has_index: bool = True,
    stale: bool = False,
    indexed_model: str | None = None,
) -> ServerState:
    """Build a ServerState matching the test scenario."""
    if indexed_model is None:
        indexed_model = DEFAULT_MODEL if has_index else None
    return ServerState(
        db_path=db_path,
        has_index=has_index,
        indexed_model_name=indexed_model,
        indexed_model_revision=None,
        configured_model_name=DEFAULT_MODEL,
        stale_model=stale,
        log_path=db_path.parent / "logs",
    )


def _insert_command(
    conn: sqlite3.Connection,
    *,
    source: str,
    text: str,
    ts: int,
    cwd: str | None = None,
    hostname: str | None = None,
    exit_code: int | None = None,
    duration_ms: int | None = None,
    session_id: str | None = None,
    text_hash: bytes | None = None,
) -> int:
    """Insert one row into commands; return the id. Tests use this to seed
    deterministic data without going through the indexer."""
    if text_hash is None:
        # Synthetic hash. Tests don't depend on it being a real BLAKE2b.
        text_hash = (text.encode("utf-8") + b"\x00" * 32)[:32]
    cur = conn.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, cwd, "
        "hostname, exit_code, duration_ms, session_id, ts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source, text, text_hash, cwd, hostname, exit_code, duration_ms, session_id, ts),
    )
    return int(cur.lastrowid or 0)


def _insert_command_with_vector(
    conn: sqlite3.Connection,
    vec: np.ndarray,
    **kwargs,
) -> int:
    """Insert a command + its sqlite-vec embedding row."""
    cmd_id = _insert_command(conn, **kwargs)
    vec_blob = np.ascontiguousarray(vec.astype(np.float32)).tobytes()
    conn.execute(
        "INSERT INTO commands_vec (command_id, embedding) VALUES (?, ?)",
        (cmd_id, vec_blob),
    )
    return cmd_id


@pytest.fixture
def db_with_index(tmp_path) -> sqlite3.Connection:
    """A migrated DB with embedding_model_name set so has_index=True."""
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)
    migrate(conn)
    set_meta(conn, "embedding_model_name", DEFAULT_MODEL)
    yield conn
    conn.close()


def _fake_unit_vec(seed: int, dim: int = 384) -> np.ndarray:
    """Deterministic L2-normalized vector for a seed integer. Same seed →
    same vector across calls (tests pin specific commands to specific
    vectors so the search ranking is reproducible)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_fake_encode(query_to_vec: dict[str, np.ndarray]):
    """Build an encode coroutine that maps known queries to fixed vectors.

    Unmapped queries return a random unit vector (so SearchInput with an
    arbitrary query doesn't crash, but the result ranking is undefined).
    """

    async def _encode(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            v = query_to_vec.get(t)
            if v is None:
                v = _fake_unit_vec(hash(t) & 0xFFFF)
            out[i] = v
        return out

    return _encode


def _run(coro):
    """Tiny helper for running async dispatch in sync tests."""
    return asyncio.run(coro)


# === Pydantic input validation ===


class TestPydanticValidation:
    def test_search_rejects_empty_query(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(query="")

    def test_search_rejects_oversized_query(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(query="x" * 1001)

    def test_search_rejects_oversized_limit(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(query="ok", limit=999)

    def test_search_rejects_zero_limit(self) -> None:
        with pytest.raises(ValidationError):
            SearchInput(query="ok", limit=0)

    def test_search_rejects_unknown_field(self) -> None:
        """extra='forbid' on the model: passing an unknown field is rejected."""
        with pytest.raises(ValidationError):
            SearchInput(query="ok", evil_field="oops")  # type: ignore[call-arg]

    def test_recent_defaults(self) -> None:
        inp = RecentInput()
        assert inp.limit == 20
        assert inp.cwd_prefix is None
        assert inp.host is None

    def test_command_stats_requires_pattern(self) -> None:
        with pytest.raises(ValidationError):
            CommandStatsInput()  # type: ignore[call-arg]

    def test_failed_recently_default_window(self) -> None:
        inp = FailedRecentlyInput()
        assert inp.window == "24h"
        assert inp.limit == 20

    def test_commands_after_default_limit(self) -> None:
        inp = CommandsAfterInput(pattern="psql")
        assert inp.limit == 10

    def test_find_in_project_query_required(self) -> None:
        with pytest.raises(ValidationError):
            FindInProjectInput()  # type: ignore[call-arg]


class TestTrailingSlashNormalization:
    """Trailing-slash normalization on cwd / cwd_prefix fields.

    Per the pre-push refinement: '/home/u/proj/' must normalize to
    '/home/u/proj' so that the prefix-boundary match anchors correctly
    on the path separator and trailing-slash inputs behave identically
    to non-trailing-slash inputs.
    """

    def test_search_cwd_prefix_normalized(self) -> None:
        assert SearchInput(query="x", cwd_prefix="/home/u/proj/").cwd_prefix == "/home/u/proj"
        assert SearchInput(query="x", cwd_prefix="/home/u/proj").cwd_prefix == "/home/u/proj"
        # Multiple trailing slashes also collapse.
        assert SearchInput(query="x", cwd_prefix="/home/u/proj///").cwd_prefix == "/home/u/proj"

    def test_root_preserved(self) -> None:
        # Root '/' is the one path where rstrip('/') returns '' — verify
        # the 'or "/"' fallback preserves it.
        assert SearchInput(query="x", cwd_prefix="/").cwd_prefix == "/"

    def test_find_in_project_cwd_normalized(self) -> None:
        assert FindInProjectInput(query="x", cwd="/etc/").cwd == "/etc"

    def test_recent_cwd_prefix_normalized(self) -> None:
        assert RecentInput(cwd_prefix="/var/log/").cwd_prefix == "/var/log"

    def test_none_passes_through(self) -> None:
        assert SearchInput(query="x").cwd_prefix is None


# === Pattern safety ===


class TestPatternSafety:
    def test_check_pattern_safe_rejects_bare_percent(self) -> None:
        with pytest.raises(ValueError, match="bare SQL wildcards"):
            _check_pattern_safe("%")

    def test_check_pattern_safe_rejects_double_percent(self) -> None:
        with pytest.raises(ValueError, match="bare SQL wildcards"):
            _check_pattern_safe("%%")

    def test_check_pattern_safe_rejects_many_percents(self) -> None:
        with pytest.raises(ValueError, match="bare SQL wildcards"):
            _check_pattern_safe("%%%%")

    def test_check_pattern_safe_allows_substring_with_percent(self) -> None:
        _check_pattern_safe("100%")  # legit usage; should not raise

    def test_check_pattern_safe_allows_normal(self) -> None:
        _check_pattern_safe("psql --host")

    def test_escape_like_handles_metachars(self) -> None:
        assert _escape_like("foo_bar") == "foo\\_bar"
        assert _escape_like("100%") == "100\\%"
        assert _escape_like("a\\b") == "a\\\\b"
        assert _escape_like("plain") == "plain"


# === _parse_since / _parse_window ===


class TestParseSince:
    def test_relative_seconds(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
        ts = _parse_since("30s", now=now)
        assert ts == int(now.timestamp()) - 30

    def test_relative_minutes(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
        ts = _parse_since("30m", now=now)
        assert ts == int(now.timestamp()) - 30 * 60

    def test_relative_hours(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
        ts = _parse_since("2h", now=now)
        assert ts == int(now.timestamp()) - 2 * 3600

    def test_relative_days(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
        ts = _parse_since("7d", now=now)
        assert ts == int(now.timestamp()) - 7 * 86400

    def test_relative_weeks(self) -> None:
        now = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
        ts = _parse_since("2w", now=now)
        assert ts == int(now.timestamp()) - 2 * 604800

    def test_absolute_iso_date(self) -> None:
        ts = _parse_since("2026-05-01")
        assert ts == int(datetime(2026, 5, 1, tzinfo=UTC).timestamp())

    def test_absolute_iso_datetime(self) -> None:
        ts = _parse_since("2026-05-01T10:00:00")
        assert ts == int(datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC).timestamp())

    def test_absolute_with_tz(self) -> None:
        ts = _parse_since("2026-05-01T10:00:00+00:00")
        assert ts == int(datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC).timestamp())

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="could not parse 'since'"):
            _parse_since("seven days")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            _parse_since("")


class TestParseWindow:
    def test_24h(self) -> None:
        assert _parse_window("24h") == 24 * 3600

    def test_7d(self) -> None:
        assert _parse_window("7d") == 7 * 86400

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="could not parse 'window'"):
            _parse_window("yesterday")

    def test_absolute_form_rejected(self) -> None:
        # _parse_window only accepts relative; ISO timestamp is wrong shape.
        with pytest.raises(ValueError, match="could not parse 'window'"):
            _parse_window("2026-05-01")


# === State errors ===


class TestStateErrors:
    def test_no_index_returns_clean_error(self, tmp_path: Path) -> None:
        state = _make_state(db_path=tmp_path / "missing.sqlite", has_index=False)
        encode = _make_fake_encode({})
        payload, count, is_err = _run(
            dispatch_tool(
                "recent",
                {"limit": 5},
                state=state,
                db_conn=None,
                encode=encode,
            )
        )
        assert is_err is True
        assert count == 0
        assert "Index not found" in payload["error"]
        # Verify the path is interpolated, not a raw '{db_path}' template.
        assert "{db_path}" not in payload["error"]
        assert str(tmp_path / "missing.sqlite") in payload["error"]

    def test_stale_model_returns_clean_error(self, tmp_path: Path, db_with_index) -> None:
        state = _make_state(
            db_path=tmp_path / "db.sqlite",
            has_index=True,
            stale=True,
            indexed_model="some/old-model",
        )
        encode = _make_fake_encode({})
        payload, _count, is_err = _run(
            dispatch_tool(
                "search",
                {"query": "anything"},
                state=state,
                db_conn=db_with_index,
                encode=encode,
            )
        )
        assert is_err is True
        assert "different embedding model" in payload["error"]
        assert "some/old-model" in payload["error"]
        assert "rebuild" in payload["error"].lower()

    def test_no_atuin_source_returns_clean_error(self, tmp_path: Path, db_with_index) -> None:
        # Insert only zsh rows; no atuin row exists.
        _insert_command(
            db_with_index,
            source="zsh",
            text="ls -la",
            ts=int(datetime.now(UTC).timestamp()),
            exit_code=0,
        )
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        encode = _make_fake_encode({})
        payload, _count, is_err = _run(
            dispatch_tool(
                "failed_recently",
                {"window": "24h"},
                state=state,
                db_conn=db_with_index,
                encode=encode,
            )
        )
        assert is_err is True
        assert "atuin" in payload["error"].lower()

    def test_cwd_context_missing_returns_clean_error(self, tmp_path: Path, db_with_index) -> None:
        """find_in_project with no cwd arg, no MCP_CLIENT_CWD, server cwd '/'."""
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        encode = _make_fake_encode({})
        # Mock os.getcwd to return '/' so the fallback chain has nothing.
        with patch("os.environ.get", return_value=None), patch("os.getcwd", return_value="/"):
            payload, _count, is_err = _run(
                dispatch_tool(
                    "find_in_project",
                    {"query": "x"},
                    state=state,
                    db_conn=db_with_index,
                    encode=encode,
                )
            )
        assert is_err is True
        assert "working-directory context" in payload["error"]

    def test_no_index_msg_function_interpolates(self, tmp_path: Path) -> None:
        msg = no_index_msg(tmp_path / "x.sqlite")
        assert str(tmp_path / "x.sqlite") in msg
        assert "{db_path}" not in msg

    def test_stale_model_msg_function_interpolates(self) -> None:
        msg = stale_model_msg("old/model", "new/model")
        assert "old/model" in msg
        assert "new/model" in msg
        assert "{indexed!r}" not in msg


# === Adversarial prefix-boundary (the marquee test) ===


class TestPrefixBoundaryMatching:
    """The cwd_prefix match must anchor on the path separator. '/home/u/proj'
    matches '/home/u/proj' itself and '/home/u/proj/sub' but NOT
    '/home/u/projector'. This is a class of bugs that bites if the SQL
    becomes ``cwd LIKE prefix || '%'`` instead of the boundary-anchored
    form. Same behavior must hold whether the user passes '/home/u/proj'
    or '/home/u/proj/' — the trailing-slash normalization in the Pydantic
    validator collapses them."""

    @pytest.fixture
    def seeded_db(self, db_with_index) -> sqlite3.Connection:
        ts = int(datetime.now(UTC).timestamp())
        _insert_command(db_with_index, source="zsh", text="cmd_in_proj", ts=ts, cwd="/home/u/proj")
        _insert_command(
            db_with_index, source="zsh", text="cmd_in_subdir", ts=ts + 1, cwd="/home/u/proj/sub"
        )
        _insert_command(
            db_with_index,
            source="zsh",
            text="cmd_in_projector",
            ts=ts + 2,
            cwd="/home/u/projector",
        )
        _insert_command(db_with_index, source="zsh", text="cmd_elsewhere", ts=ts + 3, cwd="/etc")
        return db_with_index

    @pytest.mark.parametrize("prefix", ["/home/u/proj", "/home/u/proj/", "/home/u/proj//"])
    def test_recent_prefix_includes_match_and_subdir_excludes_sibling(
        self, seeded_db, tmp_path: Path, prefix: str
    ) -> None:
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        encode = _make_fake_encode({})
        payload, count, is_err = _run(
            dispatch_tool(
                "recent",
                {"limit": 50, "cwd_prefix": prefix},
                state=state,
                db_conn=seeded_db,
                encode=encode,
            )
        )
        assert not is_err
        texts = {h["text"] for h in payload["results"]}
        assert "cmd_in_proj" in texts, f"prefix {prefix!r} should match exact dir"
        assert "cmd_in_subdir" in texts, f"prefix {prefix!r} should match subdir"
        assert "cmd_in_projector" not in texts, (
            f"prefix {prefix!r} must NOT match sibling whose name starts with the prefix"
        )
        assert "cmd_elsewhere" not in texts
        assert count == 2


# === LIKE escaping ===


class TestLikeEscaping:
    def test_command_stats_pattern_with_underscore_does_not_widen(
        self, db_with_index, tmp_path: Path
    ) -> None:
        """User searches for 'foo_bar'; should match 'foo_bar' literally,
        NOT 'fooXbar' (where '_' is the SQL LIKE 'any single char')."""
        ts = int(datetime.now(UTC).timestamp())
        _insert_command(db_with_index, source="zsh", text="run foo_bar", ts=ts)
        _insert_command(db_with_index, source="zsh", text="run fooXbar", ts=ts + 1)

        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        encode = _make_fake_encode({})
        payload, _count, is_err = _run(
            dispatch_tool(
                "command_stats",
                {"pattern": "foo_bar"},
                state=state,
                db_conn=db_with_index,
                encode=encode,
            )
        )
        assert not is_err
        assert payload["total"] == 1, "underscore should NOT widen via LIKE wildcard"


# === Handler bodies ===


class TestRecentHandler:
    def test_orders_by_ts_desc(self, db_with_index, tmp_path: Path) -> None:
        _insert_command(db_with_index, source="zsh", text="oldest", ts=100)
        _insert_command(db_with_index, source="zsh", text="middle", ts=200)
        _insert_command(db_with_index, source="zsh", text="newest", ts=300)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "recent",
                {"limit": 10},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 3
        texts = [h["text"] for h in payload["results"]]
        assert texts == ["newest", "middle", "oldest"]

    def test_filter_by_host(self, db_with_index, tmp_path: Path) -> None:
        _insert_command(db_with_index, source="zsh", text="here", ts=100, hostname="laptop")
        _insert_command(db_with_index, source="zsh", text="there", ts=200, hostname="server")
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "recent",
                {"host": "laptop"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 1
        assert payload["results"][0]["text"] == "here"

    def test_text_hash_is_16_hex_chars(self, db_with_index, tmp_path: Path) -> None:
        _insert_command(db_with_index, source="zsh", text="x", ts=100)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, _count, _ = _run(
            dispatch_tool(
                "recent",
                {"limit": 1},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        h = payload["results"][0]["text_hash"]
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    # F4 (3.13.5): hybrid ORDER BY for ts=0 fallback.
    #
    # zsh/bash readers emit ts=0 when EXTENDED_HISTORY is absent. With
    # all rows tied at ts=0, pre-fix `ORDER BY ts DESC, id ASC` collapsed
    # to id-ASC = oldest-first. The fix uses
    # `ORDER BY (CASE WHEN ts > 0 THEN 0 ELSE 1 END), ts DESC, id DESC`
    # so real-ts rows come first, then ts=0 rows by id-DESC (most-recent
    # insertion first as time-proxy).

    def test_all_ts_zero_falls_back_to_id_desc(self, db_with_index, tmp_path: Path) -> None:
        """F4 marquee test: when all ts=0 (typical zsh-only index), `recent`
        returns most-recent-INSERTED first (id DESC), NOT oldest-first.

        This is the regression-prevention test for the silent bug F4
        identified — pre-fix behavior was id ASC, contradicting the tool's
        'most-recent' semantics.
        """
        _insert_command(db_with_index, source="zsh", text="first", ts=0)
        _insert_command(db_with_index, source="zsh", text="second", ts=0)
        _insert_command(db_with_index, source="zsh", text="third", ts=0)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "recent",
                {"limit": 10},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 3
        texts = [h["text"] for h in payload["results"]]
        # id DESC (most-recently-inserted first), NOT id ASC
        assert texts == ["third", "second", "first"]
        # Off-by-one defense: explicit assertion that catches accidental
        # id ASC if a future refactor regresses the SQL.
        ids = [h["id"] for h in payload["results"]]
        assert ids[0] > ids[-1], (
            "F4 regression: results ordered id-ASC (oldest-first); "
            "expected id-DESC (most-recent insertion first) when all ts=0"
        )

    def test_mixed_ts_orders_real_ts_first_then_id_desc(
        self, db_with_index, tmp_path: Path
    ) -> None:
        """Mixed-source index: atuin rows have real ts; zsh rows have ts=0.
        Hybrid ORDER BY should put atuin rows first (by ts DESC) then zsh
        rows (by id DESC).
        """
        # Insertion order: zsh (ts=0), then atuin (ts=200), then zsh (ts=0),
        # then atuin (ts=100). Result should be:
        #   atuin ts=200 first (real-ts group, sorted DESC)
        #   atuin ts=100 second
        #   zsh row inserted later (higher id) third (ts=0 group, id DESC)
        #   zsh row inserted earlier (lower id) fourth
        _insert_command(db_with_index, source="zsh", text="zsh-old-insert", ts=0)
        _insert_command(db_with_index, source="atuin", text="atuin-newer", ts=200)
        _insert_command(db_with_index, source="zsh", text="zsh-newer-insert", ts=0)
        _insert_command(db_with_index, source="atuin", text="atuin-older", ts=100)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "recent",
                {"limit": 10},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 4
        texts = [h["text"] for h in payload["results"]]
        assert texts == [
            "atuin-newer",  # real-ts group, ts=200 first
            "atuin-older",  # real-ts group, ts=100 second
            "zsh-newer-insert",  # ts=0 group, higher id (later insertion)
            "zsh-old-insert",  # ts=0 group, lower id (earlier insertion)
        ]


class TestCommandStatsHandler:
    def test_aggregates(self, db_with_index, tmp_path: Path) -> None:
        ts = int(datetime.now(UTC).timestamp())
        _insert_command(
            db_with_index,
            source="zsh",
            text="psql -h db1",
            ts=ts,
            cwd="/proj",
            duration_ms=100,
            exit_code=0,
        )
        _insert_command(
            db_with_index,
            source="zsh",
            text="psql -h db2",
            ts=ts + 1,
            cwd="/proj",
            duration_ms=200,
            exit_code=0,
        )
        _insert_command(
            db_with_index,
            source="atuin",
            text="psql failing",
            ts=ts + 2,
            cwd="/other",
            duration_ms=50,
            exit_code=1,
        )
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "command_stats",
                {"pattern": "psql"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 3
        assert payload["total"] == 3
        # Top cwd: /proj has 2, /other has 1. Pydantic preserves tuple type
        # in model_dump(); JSON serialization at the SDK boundary converts
        # to lists (verified separately by manual smoke).
        assert payload["top_cwds"][0] == ("/proj", 2)
        # Mean duration over the 3 = (100+200+50)/3 ≈ 116.67
        assert abs(payload["mean_duration_ms"] - 350.0 / 3.0) < 0.01
        # Success: 2/3 had exit_code=0.
        assert abs(payload["success_rate"] - 2.0 / 3.0) < 1e-6
        assert payload["by_source"] == {"atuin": 1, "zsh": 2}

    def test_rejects_bare_wildcard_pattern(self, db_with_index, tmp_path: Path) -> None:
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, _count, is_err = _run(
            dispatch_tool(
                "command_stats",
                {"pattern": "%"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert is_err
        assert "bare SQL wildcards" in payload["error"]


class TestFailedRecentlyHandler:
    def test_returns_only_failures(self, db_with_index, tmp_path: Path) -> None:
        now = int(datetime.now(UTC).timestamp())
        _insert_command(db_with_index, source="atuin", text="ok", ts=now - 10, exit_code=0)
        _insert_command(db_with_index, source="atuin", text="bad1", ts=now - 100, exit_code=1)
        _insert_command(db_with_index, source="atuin", text="bad2", ts=now - 200, exit_code=2)
        # An old failure outside the window:
        _insert_command(db_with_index, source="atuin", text="old_bad", ts=now - 100000, exit_code=1)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "failed_recently",
                {"window": "1h"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 2
        texts = {h["text"] for h in payload["results"]}
        assert texts == {"bad1", "bad2"}

    def test_filter_by_pattern(self, db_with_index, tmp_path: Path) -> None:
        now = int(datetime.now(UTC).timestamp())
        _insert_command(db_with_index, source="atuin", text="psql died", ts=now, exit_code=1)
        _insert_command(db_with_index, source="atuin", text="git crashed", ts=now, exit_code=1)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "failed_recently",
                {"window": "24h", "pattern": "psql"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 1
        assert payload["results"][0]["text"] == "psql died"


class TestCommandsAfterHandler:
    def test_finds_following_commands_in_session(self, db_with_index, tmp_path: Path) -> None:
        _insert_command(
            db_with_index,
            source="atuin",
            text="run migration",
            ts=100,
            session_id="s1",
        )
        _insert_command(db_with_index, source="atuin", text="check status", ts=200, session_id="s1")
        _insert_command(db_with_index, source="atuin", text="rollback", ts=300, session_id="s1")
        # Different session — should NOT appear in 'following'.
        _insert_command(
            db_with_index,
            source="atuin",
            text="other-session-cmd",
            ts=250,
            session_id="s2",
        )
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, _ = _run(
            dispatch_tool(
                "commands_after",
                {"pattern": "migration"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert count == 1
        seq = payload["results"][0]
        assert seq["pattern_match"]["text"] == "run migration"
        following_texts = [c["text"] for c in seq["following"]]
        assert following_texts == ["check status", "rollback"]

    def test_no_atuin_source_returns_clean_error(self, db_with_index, tmp_path: Path) -> None:
        """F1 (3.13.5): commands_after requires atuin source data for
        session-boundary tracking. Non-atuin index returns a clean state-
        error with the user-facing message; the message surfaces the
        correctness reason (cross-session conflation).

        Pre-F1 behavior was silent UX degradation (matched pattern,
        empty `following`); replaced by explicit error surface so the
        LLM client + user understand why no sequence data is available.
        """
        # Only zsh rows — no atuin records in the index.
        _insert_command(db_with_index, source="zsh", text="orphan_cmd", ts=100, session_id=None)
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, is_err = _run(
            dispatch_tool(
                "commands_after",
                {"pattern": "orphan_cmd"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert is_err is True
        assert count == 0
        msg = payload["error"]
        assert "commands_after requires atuin" in msg
        # Locked message text includes the cross-session-conflation
        # parenthetical surfacing the correctness reason
        assert "conflate commands across concurrent terminal sessions" in msg
        assert "https://atuin.sh" in msg


class TestSearchHandler:
    def test_semantic_topk_with_fixture_vectors(self, db_with_index, tmp_path: Path) -> None:
        """Pin specific commands to specific vectors. The query vector is
        the SAME as one of the corpus vectors — that command must rank
        first (cosine sim 1.0)."""
        v1 = _fake_unit_vec(seed=1)
        v2 = _fake_unit_vec(seed=2)
        v3 = _fake_unit_vec(seed=3)
        _insert_command_with_vector(db_with_index, v1, source="zsh", text="match-target", ts=100)
        _insert_command_with_vector(db_with_index, v2, source="zsh", text="other-1", ts=200)
        _insert_command_with_vector(db_with_index, v3, source="zsh", text="other-2", ts=300)

        encode = _make_fake_encode({"give-me-target": v1})
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, count, is_err = _run(
            dispatch_tool(
                "search",
                {"query": "give-me-target", "limit": 3},
                state=state,
                db_conn=db_with_index,
                encode=encode,
            )
        )
        assert not is_err
        assert count == 3
        # Top result must be the matching command with score ~1.0.
        top = payload["results"][0]
        assert top["text"] == "match-target"
        assert top["score"] > 0.999

    def test_search_with_invalid_since_returns_error(self, db_with_index, tmp_path: Path) -> None:
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, _count, is_err = _run(
            dispatch_tool(
                "search",
                {"query": "x", "since": "yesterday"},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        assert is_err
        assert "could not parse 'since'" in payload["error"]


class TestFindInProjectCwdResolution:
    def test_explicit_cwd_wins(self) -> None:
        with patch.dict("os.environ", {"MCP_CLIENT_CWD": "/from-env"}):
            result = _resolve_find_in_project_cwd("/explicit")
        assert result == "/explicit"

    def test_env_falls_back(self) -> None:
        with patch.dict("os.environ", {"MCP_CLIENT_CWD": "/from-env/"}, clear=False):
            result = _resolve_find_in_project_cwd(None)
        # Trailing slash should be normalized.
        assert result == "/from-env"

    def test_server_startup_cwd_last(self, tmp_path: Path) -> None:
        # Ensure no env var, server cwd is a real path.
        env = {k: v for k, v in __import__("os").environ.items() if k != "MCP_CLIENT_CWD"}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("os.getcwd", return_value=str(tmp_path)),
        ):
            result = _resolve_find_in_project_cwd(None)
        assert result == str(tmp_path)

    def test_root_returns_none(self) -> None:
        env = {k: v for k, v in __import__("os").environ.items() if k != "MCP_CLIENT_CWD"}
        with patch.dict("os.environ", env, clear=True), patch("os.getcwd", return_value="/"):
            result = _resolve_find_in_project_cwd(None)
        assert result is None


class TestDefenseInDepthScrub:
    def test_output_text_is_scrubbed(self, db_with_index, tmp_path: Path) -> None:
        """Even if a row in the DB somehow contains a secret-shaped string
        (the indexer would have already scrubbed at write time, but this
        is the second layer of defense), the tool output must scrub before
        returning. Per CLAUDE.md §1: 'Defense in depth: scrub at index
        time AND at query response.'"""
        # Force-insert a row containing a secret-shaped string. The
        # scrubber should rewrite it at output time.
        token = "abc123XYZ_long_token_value"
        text = f"curl https://api.example.com -H 'Authorization: Bearer {token}'"
        _insert_command(
            db_with_index,
            source="zsh",
            text=text,
            ts=100,
        )
        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        payload, _count, _ = _run(
            dispatch_tool(
                "recent",
                {"limit": 1},
                state=state,
                db_conn=db_with_index,
                encode=_make_fake_encode({}),
            )
        )
        out_text = payload["results"][0]["text"]
        # Scrubber should have wrapped the bearer token.
        assert token not in out_text
        assert "<REDACTED:" in out_text


# === Tools registry ===


class TestToolsRegistry:
    def test_six_tools_registered(self) -> None:
        from recall.tools import TOOLS

        names = [t.name for t in TOOLS]
        assert names == [
            "search",
            "find_in_project",
            "commands_after",
            "failed_recently",
            "command_stats",
            "recent",
        ]

    def test_each_tool_has_input_schema(self) -> None:
        from recall.tools import TOOLS

        for t in TOOLS:
            assert t.inputSchema is not None
            assert "properties" in t.inputSchema, f"{t.name} missing properties"

    def test_each_tool_schema_includes_required_key(self) -> None:
        """F2 (3.13.5) shim: every tool's JSON Schema must include the
        ``required`` key (even if []), not omit it.

        Pydantic 2 omits the key when no fields are required; some MCP
        clients filter tools whose schema lacks the declaration. The
        shim in `_tool_for()` sets it explicitly. This test asserts the
        contract.
        """
        from recall.tools import TOOLS

        for t in TOOLS:
            assert "required" in t.inputSchema, (
                f"{t.name} schema missing 'required' key — F2 shim broken"
            )
            assert isinstance(t.inputSchema["required"], list), f"{t.name} 'required' is not a list"
        # Sanity: at least one tool with required parameters (search has
        # 'query'); at least one without (recent has all defaults). The
        # shim affects the no-required-params case specifically.
        recent_schema = next(t.inputSchema for t in TOOLS if t.name == "recent")
        assert recent_schema["required"] == [], (
            "recent has no required params; shim should produce []"
        )
        search_schema = next(t.inputSchema for t in TOOLS if t.name == "search")
        assert search_schema["required"] == ["query"], (
            "search's required params should be unchanged by the shim"
        )

    def test_descriptions_are_non_empty(self) -> None:
        from recall.tools import TOOLS

        for t in TOOLS:
            assert t.description and len(t.description) > 20, t.name

    def test_unknown_tool_raises_mcp_error(self, db_with_index, tmp_path: Path) -> None:
        from mcp.shared.exceptions import McpError

        state = _make_state(db_path=tmp_path / "db.sqlite", has_index=True)
        with pytest.raises(McpError) as exc:
            _run(
                dispatch_tool(
                    "nonexistent_tool",
                    {},
                    state=state,
                    db_conn=db_with_index,
                    encode=_make_fake_encode({}),
                )
            )
        assert "unknown tool" in exc.value.error.message.lower()
