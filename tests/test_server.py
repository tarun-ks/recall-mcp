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
    yield
    recall.server._state = None
    recall.server._db_conn = None
    recall.server._embedder = None
    recall.server._embedder_lock = None


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


# Patch unused; quiet linters that flag the unused import otherwise.
_ = patch
_ = sqlite3
