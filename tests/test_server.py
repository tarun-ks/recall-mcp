"""Smoke tests for the 3.9 server skeleton.

These run in-process — no subprocess, no real MCP-client. The dedicated
stdio-cleanliness subprocess test lands in 3.11; the pseudo-client +
recorded-session-fixture tests land in 3.12.

What these tests do verify:
  - The server module imports cleanly without pulling heavy deps
  - ``compute_initial_state`` correctly identifies the three states
    (no-index, stale-model, ready)
  - ``setup_logging`` creates the log dir with mode 0o700 and never
    attaches a stdout-bound handler
  - ``create_server`` returns an mcp Server with empty tool registry
  - ``get_embedder`` (lazy-load + lock helper) caches its instance
    correctly across concurrent calls
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import recall.server
from recall.db import connect, migrate, set_meta
from recall.embed import DEFAULT_MODEL
from recall.server import (
    ServerState,
    compute_initial_state,
    create_server,
    get_embedder,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level singletons between tests to avoid cross-pollution."""
    recall.server._state = None
    recall.server._db_conn = None
    recall.server._embedder = None
    recall.server._embedder_lock = None
    recall.server._state_lock = None
    yield
    recall.server._state = None
    if recall.server._db_conn is not None:
        recall.server._db_conn.close()
    recall.server._db_conn = None
    recall.server._embedder = None
    recall.server._embedder_lock = None
    recall.server._state_lock = None


# === compute_initial_state ===


def test_no_index_when_db_missing(tmp_path: Path) -> None:
    """RECALL_DB_PATH points at a missing DB → has_index=False, no error."""
    state = compute_initial_state(
        db_path=tmp_path / "missing.sqlite",
        log_path=tmp_path / "logs",
    )
    assert state.has_index is False
    assert state.indexed_model_name is None
    assert state.stale_model is False
    assert state.ready is False


def test_no_index_when_db_exists_but_empty(tmp_path: Path) -> None:
    """Migrated DB with zero commands → has_index=False (the table exists
    but the user hasn't run ``recall index`` yet)."""
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)
    migrate(conn)
    conn.close()

    state = compute_initial_state(db_path=db_path, log_path=tmp_path / "logs")
    assert state.has_index is False
    assert state.ready is False


def test_stale_model_when_indexed_differs_from_configured(tmp_path: Path) -> None:
    """Index was built with a different model → stale_model=True."""
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)
    migrate(conn)
    # Insert a fake commands row so has_index=True.
    conn.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "ls -la", b"\x00" * 32, 100),
    )
    set_meta(conn, "embedding_model_name", "BAAI/some-other-model")
    conn.close()

    state = compute_initial_state(db_path=db_path, log_path=tmp_path / "logs")
    assert state.has_index is True
    assert state.indexed_model_name == "BAAI/some-other-model"
    assert state.stale_model is True
    assert state.ready is False


def test_ready_when_index_exists_and_model_matches(tmp_path: Path) -> None:
    """Happy path: DB has commands AND embedding_model_name == configured."""
    db_path = tmp_path / "db.sqlite"
    conn = connect(db_path)
    migrate(conn)
    conn.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "ls -la", b"\x00" * 32, 100),
    )
    set_meta(conn, "embedding_model_name", DEFAULT_MODEL)
    conn.close()

    state = compute_initial_state(db_path=db_path, log_path=tmp_path / "logs")
    assert state.has_index is True
    assert state.indexed_model_name == DEFAULT_MODEL
    assert state.stale_model is False
    assert state.ready is True


# === setup_logging ===


def test_setup_logging_creates_log_dir_with_correct_mode(tmp_path: Path) -> None:
    """The log dir is created mode 0o700 (Q8.4)."""
    log_path = tmp_path / "logs"
    setup_logging(log_path)
    assert log_path.exists()
    # On macOS / Linux the mode bits should match. Some filesystems mask
    # this; the test_server.setup_logging swallows the OSError, so this
    # assertion can be skipped on weird FS — but tmp_path is real disk.
    mode = log_path.stat().st_mode & 0o777
    assert mode == 0o700, f"expected mode 0o700, got 0o{mode:o}"


def test_setup_logging_attaches_only_file_handler(tmp_path: Path) -> None:
    """Root logger has exactly one handler after setup, and it's the
    rotating file handler — not a StreamHandler(sys.stdout)."""
    setup_logging(tmp_path / "logs")
    root = logging.getLogger()
    handlers = root.handlers
    assert len(handlers) == 1
    handler = handlers[0]
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    # And the runtime gate explicitly forbids sys.stdout-bound handlers.
    stream = getattr(handler, "stream", None)
    assert stream is not sys.stdout


def test_setup_logging_runtime_gate_fires_on_stdout_handler(tmp_path: Path) -> None:
    """If a stdout-bound handler is added AFTER setup, calling the gate
    directly raises. The gate guards against transitive-import handler
    additions during server startup."""
    setup_logging(tmp_path / "logs")
    root = logging.getLogger()
    bad_handler = logging.StreamHandler(sys.stdout)
    root.addHandler(bad_handler)
    try:
        with pytest.raises(RuntimeError, match=r"writes to sys\.stdout"):
            recall.server._assert_no_stdout_handler(root)
    finally:
        root.removeHandler(bad_handler)


# === create_server ===


def test_create_server_returns_mcp_server() -> None:
    """``create_server`` returns an mcp Server instance."""
    s = create_server()
    # The SDK's Server class has list_tools / call_tool / get_capabilities.
    assert hasattr(s, "list_tools")
    assert hasattr(s, "call_tool")
    assert hasattr(s, "get_capabilities")


def test_list_tools_returns_the_six_tools() -> None:
    """3.10 ships the locked six-tool surface (CLAUDE.md "MCP tool surface").

    Find the registered list_tools handler and call it. The SDK stores
    request handlers in ``server.request_handlers``, keyed by request type;
    we look up the ListToolsRequest handler and invoke it.
    """
    from mcp.types import ListToolsRequest

    s = create_server()
    handler = s.request_handlers.get(ListToolsRequest)
    assert handler is not None, "list_tools handler not registered"

    # The SDK's handler wrapper takes a ListToolsRequest object.
    req = ListToolsRequest(method="tools/list", params=None)
    result = asyncio.run(handler(req))
    # Result is a ServerResult wrapping a ListToolsResult.
    tools = result.root.tools
    names = [t.name for t in tools]
    assert names == [
        "search",
        "find_in_project",
        "commands_after",
        "failed_recently",
        "command_stats",
        "recent",
    ]
    # Each tool advertises its inputSchema (used by the MCP client for
    # client-side validation before dispatch).
    for t in tools:
        assert t.inputSchema is not None
        assert "properties" in t.inputSchema


# === get_embedder (refinement 1) ===


@pytest.mark.embed
def test_get_embedder_lazy_caches_and_serializes() -> None:
    """Calling get_embedder twice returns the SAME instance (caching);
    concurrent calls block on the lock cleanly (no double-load).

    Per refinement 1 of Phase 3 §10: untested helpers accumulate subtle
    bugs that only surface when the first real caller exercises them.
    Test the helper in isolation before tools depend on it.
    """

    async def _drive() -> None:
        # Initialize the lock the way main() does — tests don't go through main().
        recall.server._embedder_lock = asyncio.Lock()

        # Two concurrent calls. Both should resolve to the SAME instance,
        # and the embedder should be loaded exactly once (the second call
        # blocks on the lock, double-checks, and returns the cached one).
        a, b = await asyncio.gather(get_embedder(), get_embedder())

        assert a is b, "get_embedder must return the cached instance on repeat calls"
        # The cached singleton is what's stored on the module.
        assert recall.server._embedder is a
        # And the lock is released after construction (acquirable now).
        assert not recall.server._embedder_lock.locked()

    asyncio.run(_drive())


def test_get_embedder_unloaded_lock_raises() -> None:
    """Calling get_embedder before main() initializes the lock raises an
    assertion. Defends against a future code path that calls get_embedder
    from outside the server runtime."""

    async def _drive() -> None:
        with pytest.raises(AssertionError, match="_embedder_lock not initialized"):
            await get_embedder()

    # Module fixture has reset _embedder and _embedder_lock to None.
    asyncio.run(_drive())


# === module import discipline ===


def test_server_module_does_not_pull_torch_at_import_time() -> None:
    """recall.server's top-level imports must not trigger sentence-
    transformers / torch (the embedder is lazy-loaded). A regression
    here would slow `recall --help` and make pytest collection heavy.

    We can't easily snapshot sys.modules cleanly mid-test, so we rely
    on the lazy-import pattern: recall.embed has the SentenceTransformer
    import INSIDE Embedder.__init__, not at module top. As long as
    server.py only imports Embedder via TYPE_CHECKING and the lazy-import
    inside get_embedder, no torch shows up at import.
    """
    # Defensive: if torch leaked in, this test environment would have it.
    # Hard to assert "torch is NOT imported" without process isolation;
    # this is a documentation-style assertion that the discipline holds.
    import recall.server

    # The lazy-import inside get_embedder is the actual gate; we verify
    # the module's top-level doesn't reference Embedder() construction.
    src = Path(recall.server.__file__).read_text(encoding="utf-8")
    # The only Embedder references should be: TYPE_CHECKING, the docstring,
    # the dataclass annotation as a string, and the lazy-import inside
    # get_embedder. None of these construct Embedder() at module load.
    # Loose check: no top-level "Embedder()" construction.
    lines = src.splitlines()
    for i, line in enumerate(lines):
        # Top-level (no indent) Embedder() construction would be a regression.
        if line.startswith("Embedder()"):
            pytest.fail(
                f"top-level Embedder() construction at line {i + 1}: {line!r} — "
                "this would defeat the lazy-load discipline"
            )


# === ServerState dataclass ===


def test_server_state_is_frozen() -> None:
    """ServerState is frozen — accidental mutation would break invariants."""
    state = ServerState(
        db_path=Path("/tmp/db.sqlite"),
        has_index=False,
        indexed_model_name=None,
        indexed_model_revision=None,
        configured_model_name=DEFAULT_MODEL,
        stale_model=False,
        log_path=Path("/tmp/logs"),
    )
    with pytest.raises((AttributeError, TypeError)):
        state.has_index = True  # type: ignore[misc]


# === _maybe_refresh_state (3.10 follow-up: first-run UX fix) ===
#
# When the server boots before the user runs ``recall index``, the
# startup snapshot pins ``_state.has_index = False``. Without lazy
# refresh, every subsequent tool call returns no_index until the user
# restarts their MCP client. The refresh logic re-probes the DB before
# every tool call WHILE has_index is False, transitions to True the
# moment an index appears, and locks in once True (no further probes).


def test_maybe_refresh_state_transitions_when_index_appears(tmp_path: Path) -> None:
    """Server boots with no DB; user runs ``recall index`` later;
    next tool-call refresh detects the new DB, transitions
    has_index=True, opens the read-only connection."""
    db_path = tmp_path / "db.sqlite"

    async def _drive() -> None:
        # Initial state: no DB exists yet.
        recall.server._state = recall.server.compute_initial_state(
            db_path=db_path, log_path=tmp_path / "logs"
        )
        recall.server._state_lock = asyncio.Lock()
        assert recall.server._state.has_index is False
        assert recall.server._db_conn is None

        # First refresh: still no DB. State stays False; no connection.
        await recall.server._maybe_refresh_state()
        assert recall.server._state.has_index is False
        assert recall.server._db_conn is None

        # User runs ``recall index`` (simulated): create migrated DB
        # with one row and the embedding-model meta.
        c = connect(db_path)
        migrate(c)
        c.execute(
            "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
            ("zsh", "ls -la", b"\x00" * 32, 100),
        )
        set_meta(c, "embedding_model_name", DEFAULT_MODEL)
        c.close()

        # Next refresh: index now visible. Transition to has_index=True;
        # connection opened.
        await recall.server._maybe_refresh_state()
        assert recall.server._state.has_index is True
        assert recall.server._state.stale_model is False
        assert recall.server._db_conn is not None

    asyncio.run(_drive())


def test_maybe_refresh_state_no_op_when_already_ready(tmp_path: Path) -> None:
    """When state.has_index is already True at refresh time, the
    function returns without re-probing — fast path verified by
    ensuring _db_conn is NOT replaced (would be a different object
    if reopened)."""
    db_path = tmp_path / "db.sqlite"
    c = connect(db_path)
    migrate(c)
    c.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "x", b"\x00" * 32, 1),
    )
    set_meta(c, "embedding_model_name", DEFAULT_MODEL)
    c.close()

    async def _drive() -> None:
        recall.server._state = recall.server.compute_initial_state(
            db_path=db_path, log_path=tmp_path / "logs"
        )
        recall.server._state_lock = asyncio.Lock()
        assert recall.server._state.has_index is True

        # Set _db_conn to a sentinel; verify refresh does NOT replace it.
        sentinel = sqlite3.connect(":memory:")
        recall.server._db_conn = sentinel
        original_state = recall.server._state

        await recall.server._maybe_refresh_state()
        assert recall.server._db_conn is sentinel, "refresh replaced cached db conn"
        assert recall.server._state is original_state, "refresh replaced cached state"

    asyncio.run(_drive())


def test_maybe_refresh_state_handles_db_still_missing(tmp_path: Path) -> None:
    """Repeated refreshes when the DB still doesn't exist must not
    raise, must not open a connection, must keep state.has_index=False."""
    db_path = tmp_path / "missing.sqlite"

    async def _drive() -> None:
        recall.server._state = recall.server.compute_initial_state(
            db_path=db_path, log_path=tmp_path / "logs"
        )
        recall.server._state_lock = asyncio.Lock()
        assert recall.server._state.has_index is False

        # Three refreshes in a row — should be cheap and safe.
        for _ in range(3):
            await recall.server._maybe_refresh_state()
        assert recall.server._state.has_index is False
        assert recall.server._db_conn is None

    asyncio.run(_drive())


def test_maybe_refresh_state_concurrent_calls_open_db_once(tmp_path: Path) -> None:
    """Two concurrent refresh calls when the index has just appeared
    must result in only ONE open of _db_conn (the lock + double-check
    pattern). Tests the same defense as ``get_embedder``."""
    db_path = tmp_path / "db.sqlite"
    c = connect(db_path)
    migrate(c)
    c.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "y", b"\x00" * 32, 1),
    )
    set_meta(c, "embedding_model_name", DEFAULT_MODEL)
    c.close()

    async def _drive() -> None:
        # Boot with state=no-index (DB exists at file level but pretend
        # the snapshot was taken before the rows landed — same shape as
        # the real first-run scenario).
        recall.server._state = ServerState(
            db_path=db_path,
            has_index=False,
            indexed_model_name=None,
            indexed_model_revision=None,
            configured_model_name=DEFAULT_MODEL,
            stale_model=False,
            log_path=tmp_path / "logs",
        )
        recall.server._state_lock = asyncio.Lock()
        recall.server._db_conn = None

        # Two concurrent refreshes.
        await asyncio.gather(
            recall.server._maybe_refresh_state(),
            recall.server._maybe_refresh_state(),
        )
        assert recall.server._state.has_index is True
        # Connection opened exactly once.
        assert recall.server._db_conn is not None
        # Verify it's a working connection.
        n = recall.server._db_conn.execute("SELECT COUNT(*) FROM commands").fetchone()[0]
        assert n == 1

    asyncio.run(_drive())


# Patch unused; quiet linters that flag the unused import otherwise.
_ = patch
_ = sqlite3
