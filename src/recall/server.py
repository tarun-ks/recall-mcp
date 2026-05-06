"""recall MCP server — stdio transport.

Phase 3 lands the MCP server in five commits:

    3.9  Server skeleton: lifecycle, DB connection, embedder lazy-load,
         stale-index check, empty tool registry. (this commit)
    3.10 Six tool implementations + Pydantic input/output schemas.
    3.11 stdio cleanliness test suite (subprocess + ruff T201 + AST + runtime).
    3.12 Pseudo-client + recorded-session-fixture replay tests.
    3.13 Manual smoke-test checklist + docs/clients-tested.md.

ARCHITECTURE (locked in Phase-3 §§1-10 planning round)

  Q1 SDK:        official ``mcp`` package, pinned ``>=1.0.0,<2.0.0``;
                 SDK's built-in ``stdio_server`` runner (no wrapper).
  Q4 async:      ``asyncio.to_thread`` per encode call;
                 ``asyncio.Lock`` serializes encodes (MPS thread-safety).
                 Shared read-only DB connection at startup.
  Q5 errors:     Strategy (a) — server starts even with no/stale index;
                 tool calls return structured MCP errors per state.
  Q7 lifecycle:  Lazy embedder load (first tool call pays ~5s).
                 Startup-only stale-index detection.
                 Concurrent indexer+server is unsupported (documented).
  Q8 logging:    INFO default; ``RotatingFileHandler`` at
                 ``~/.recall/logs/recall.log`` (mode 0o700);
                 query text NOT logged by default.

THE STDIO CLEANLINESS INVARIANT (CLAUDE.md §5)

  After ``stdio_server()`` opens the protocol loop, NO byte may reach
  stdout except an MCP JSON-RPC frame. The dedicated test suite lands
  in 3.11; this module's discipline:

  - All logging goes to file or stderr — never stdout.
  - No ``print()`` calls anywhere in the module.
  - The runtime defense (redirect_stdout context manager around
    embedder.encode) lands in 3.10 with the first tool that uses
    embeddings.
  - ``setup_logging`` asserts no handler points at sys.stdout
    before returning.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import logging.handlers
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mcp.server.stdio
from mcp.server import InitializationOptions, NotificationOptions, Server
from mcp.types import Tool

from recall.db import DBError, _default_db_path, connect_readonly, get_meta
from recall.embed import DEFAULT_MODEL

if TYPE_CHECKING:
    from recall.embed import Embedder

_LOG = logging.getLogger("recall.server")

_SERVER_NAME = "recall-mcp"
_SERVER_VERSION = "0.0.1"


@dataclass(frozen=True)
class ServerState:
    """Snapshot of server-relevant state computed at startup.

    Used by tool handlers (3.10+) to surface clean structured errors
    via the MCP protocol when the index is missing or stale, instead
    of failing the tool call with a stack trace.

    Per Q5.1 strategy (a): the server starts up successfully even when
    the index doesn't exist or the model is stale; each tool call sees
    the state and returns the appropriate error.
    """

    db_path: Path
    has_index: bool  # commands table exists + has rows
    indexed_model_name: str | None  # meta.embedding_model_name; None if no index
    indexed_model_revision: str | None  # meta.embedding_model_revision; "" or None
    configured_model_name: str  # recall.embed.DEFAULT_MODEL
    stale_model: bool  # has_index AND indexed_model_name != configured_model_name
    log_path: Path

    @property
    def ready(self) -> bool:
        """True when the server can serve queries — index exists and model matches."""
        return self.has_index and not self.stale_model


# Module-level state for the server process. Per-process singletons; the
# server is single-tenant by design (one MCP client connection at a time
# via stdio). State is computed once at startup; tool handlers read it.
_state: ServerState | None = None
_db_conn: sqlite3.Connection | None = None
_embedder: Embedder | None = None
_embedder_lock: asyncio.Lock | None = None  # constructed in main()


def compute_initial_state(
    db_path: Path,
    log_path: Path,
    configured_model: str = DEFAULT_MODEL,
) -> ServerState:
    """Compute the startup state snapshot. Pure function modulo DB read.

    Probes the DB at ``db_path`` (may not exist; that's a state, not an
    error). Reads ``meta.embedding_model_name`` if the DB exists and is
    migrated. Compares against ``configured_model``.
    """
    if not db_path.exists():
        return ServerState(
            db_path=db_path,
            has_index=False,
            indexed_model_name=None,
            indexed_model_revision=None,
            configured_model_name=configured_model,
            stale_model=False,
            log_path=log_path,
        )

    # DB exists; probe it. We open a separate read-only connection just
    # for the state probe (the long-lived ``_db_conn`` is opened in main()
    # after this returns). Closing the probe connection avoids leaving
    # a stray FD around.
    has_index = False
    indexed_model_name: str | None = None
    indexed_model_revision: str | None = None
    try:
        probe = connect_readonly(db_path)
        try:
            # has_index: commands table exists AND non-empty.
            row = probe.execute("SELECT COUNT(*) FROM commands").fetchone()
            has_index = int(row[0]) > 0
            indexed_model_name = get_meta(probe, "embedding_model_name")
            indexed_model_revision = get_meta(probe, "embedding_model_revision")
        finally:
            probe.close()
    except (sqlite3.OperationalError, DBError):
        # Table missing → no index. Treat as state, not error.
        pass

    stale_model = has_index and (indexed_model_name != configured_model)

    return ServerState(
        db_path=db_path,
        has_index=has_index,
        indexed_model_name=indexed_model_name,
        indexed_model_revision=indexed_model_revision or None,
        configured_model_name=configured_model,
        stale_model=stale_model,
        log_path=log_path,
    )


def setup_logging(log_path: Path, level: int = logging.INFO) -> None:
    """Configure the recall.server logger.

    File handler: ``RotatingFileHandler`` at ``log_path/recall.log``,
    10 MB × 3 backups (Q8.3). Directory created with mode 0o700 (Q8.4).

    NEVER attaches a handler that writes to ``sys.stdout`` — the MCP
    stdio transport reserves stdout for JSON-RPC frames.

    Asserts at the end that no handler in the root logger writes to
    ``sys.stdout``. This is the runtime gate — if a future change
    accidentally adds a stdout-bound handler, server startup fails fast.
    """
    log_path.mkdir(parents=True, exist_ok=True)
    # chmod can fail on weird filesystems (network shares, sandboxes);
    # not fatal — the log dir exists, that's what matters.
    with contextlib.suppress(OSError):
        os.chmod(log_path, 0o700)

    log_file = log_path / "recall.log"

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Replace any existing handlers — we want exclusive control of where
    # log records go. A stray StreamHandler(sys.stdout) from a transitive
    # import would silently break MCP framing.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)

    # Belt-and-suspenders runtime gate.
    _assert_no_stdout_handler(root)


def _assert_no_stdout_handler(logger: logging.Logger) -> None:
    """Walk a logger's handlers; raise if any writes to ``sys.stdout``."""
    for h in logger.handlers:
        stream = getattr(h, "stream", None)
        if stream is sys.stdout:
            raise RuntimeError(
                f"recall.server: logging handler {h!r} writes to sys.stdout. "
                "MCP stdio transport requires stdout to carry only JSON-RPC "
                "frames. Reconfigure logging to use stderr or a file."
            )


async def get_embedder() -> Embedder:
    """Lazy-load the Embedder; cache for the server process lifetime.

    First call pays ~5s (model load + MPS warmup); subsequent calls
    return the cached instance. The double-checked-locking pattern
    handles concurrent first calls cleanly.

    Per Q4.2: ``_embedder_lock`` also serializes encode calls in the
    ``encode_async`` helper that lands 3.10 — this function only
    handles construction.
    """
    global _embedder
    if _embedder is not None:
        return _embedder
    assert _embedder_lock is not None, "_embedder_lock not initialized; call main() first"
    async with _embedder_lock:
        if _embedder is None:  # double-check after lock
            from recall.embed import Embedder as _Embedder

            _LOG.info("recall.server: lazy-loading embedder (first tool call)")
            _embedder = await asyncio.to_thread(_Embedder)
            _LOG.info(
                "recall.server: embedder ready (model=%s, dim=%d)",
                _embedder.model_name,
                _embedder.dim,
            )
    assert _embedder is not None
    return _embedder


def create_server() -> Server:
    """Construct the MCP Server with handlers registered.

    3.9 ships an empty tool registry — ``list_tools`` returns ``[]``.
    3.10 adds the six tools via ``@server.call_tool`` decorators on
    handler functions.
    """
    server: Server = Server(_SERVER_NAME, version=_SERVER_VERSION)

    # mcp SDK 1.27 doesn't fully type its decorator-based handler-registration
    # pattern; the decorator factory returns Any. This is correct SDK usage
    # — the type-ignores quiet mypy until the SDK ships better stubs.
    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def _list_tools() -> list[Tool]:
        # 3.10 will populate this with the six tools.
        return []

    return server


async def main() -> None:
    """Entry point invoked from the ``recall serve`` CLI command.

    Sets up state + DB + lock; constructs the Server; runs the SDK's
    stdio loop. The CLI configures logging BEFORE calling main() so
    no startup byte ever reaches stdout (per the stdio-cleanliness
    invariant in this module's docstring).
    """
    global _state, _db_conn, _embedder_lock

    db_path = _default_db_path()
    log_path = Path.home() / ".recall" / "logs"

    _state = compute_initial_state(db_path=db_path, log_path=log_path)
    _LOG.info(
        "recall.server: state computed db=%s has_index=%s stale_model=%s",
        _state.db_path,
        _state.has_index,
        _state.stale_model,
    )

    # Open the long-lived read-only DB connection IF the index exists.
    # When has_index is False, _db_conn stays None; tool handlers (3.10)
    # check _state.has_index and return a structured error before
    # touching _db_conn.
    if _state.has_index:
        _db_conn = connect_readonly(db_path)
        _LOG.info("recall.server: db connection opened (read-only)")

    _embedder_lock = asyncio.Lock()

    server = create_server()

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=_SERVER_NAME,
                server_version=_SERVER_VERSION,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


__all__ = (
    "ServerState",
    "compute_initial_state",
    "create_server",
    "get_embedder",
    "main",
    "setup_logging",
)
