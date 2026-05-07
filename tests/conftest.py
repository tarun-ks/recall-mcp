"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.fixtures.make_atuin_fixture import make_atuin_fixture


@pytest.fixture(scope="session")
def atuin_fixture_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a deterministic atuin DB once per test session and return its path.

    See ``tests/fixtures/make_atuin_fixture.py`` for the row contents.
    Fixture lives under ``tmp_path`` so we never commit a binary DB.
    """
    path = tmp_path_factory.mktemp("atuin") / "history.db"
    make_atuin_fixture(path)
    return path


# === MCP subprocess session helper (3.11) ===
#
# Spawns ``recall serve`` in a real subprocess, holds stdin open, sends MCP
# JSON-RPC frames one at a time as line-delimited JSON, reads responses on
# stdout, captures stderr to a tmpfile. Designed for reuse:
#
#   - 3.11 stdio-cleanliness tests use this directly.
#   - 3.12's pseudo-client + recorded-session-fixture replay tests inherit
#     the same helper API (initialize / list_tools / call_tool / close /
#     stderr_text) without modification per the locked plan.
#
# If 3.12 needs API adjustments (recorded-session replay might want
# generic ``send_request(method, params)`` for non-tool requests), surface
# them explicitly during 3.12's planning round so we know whether 3.11's
# tests need re-running against the new API or whether 3.12 inherits as-is.


class MCPSubprocessSession:
    """Async helper around ``recall serve`` running in a subprocess.

    Uses line-delimited JSON over stdin/stdout, matching the MCP stdio
    transport's wire format. Holds stdin open across all sends — this is
    the fix for the 3.10 static-pipe-EOF finding (CLAUDE.md §5).
    """

    def __init__(self, proc: subprocess.Popen[bytes], stderr_path: Path) -> None:
        self._proc = proc
        self._stderr_path = stderr_path
        self._next_id = 1
        # Read responses asynchronously so concurrent send/recv doesn't
        # deadlock on a slow server. asyncio's StreamReader doesn't wrap
        # Popen.stdout cleanly; we use a small read-line helper instead.
        assert proc.stdout is not None
        assert proc.stdin is not None
        self._stdout = proc.stdout
        self._stdin = proc.stdin

    async def _read_line(self, *, timeout: float = 30.0) -> bytes:
        """Read one line from the subprocess's stdout. Times out.

        Per-line read because MCP stdio framing is one JSON object per
        line (no Content-Length headers like LSP's framed mode).
        """
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, self._stdout.readline)
        line = await asyncio.wait_for(future, timeout=timeout)
        if not line:
            # EOF — server died. Capture stderr for the failure message.
            stderr = self.stderr_text()
            raise RuntimeError(f"recall serve closed stdout unexpectedly. stderr:\n{stderr}")
        return line

    async def _send(self, frame: dict[str, Any]) -> None:
        data = (json.dumps(frame) + "\n").encode("utf-8")
        self._stdin.write(data)
        self._stdin.flush()

    async def initialize(self) -> dict[str, Any]:
        """Send initialize, return parsed result dict."""
        req_id = self._next_id
        self._next_id += 1
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-subprocess", "version": "0.0.1"},
                },
            }
        )
        line = await self._read_line()
        return json.loads(line)

    async def notify_initialized(self) -> None:
        """Send the initialized notification (no response expected)."""
        await self._send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )

    async def list_tools(self) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        await self._send({"jsonrpc": "2.0", "id": req_id, "method": "tools/list"})
        line = await self._read_line()
        return json.loads(line)

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send tools/call; return the parsed JSON-RPC response.

        Custom timeout for tool calls — semantic search with a cold-cache
        embedder pays ~25s on Linux CPU. Default 30s; cold-cache tests
        override.
        """
        req_id = self._next_id
        self._next_id += 1
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        line = await self._read_line(timeout=timeout)
        return json.loads(line)

    async def close(self) -> None:
        """Close stdin (signals EOF), then wait briefly for graceful exit
        before SIGTERM → SIGKILL escalation."""
        import contextlib

        with contextlib.suppress(BrokenPipeError, OSError):
            self._stdin.close()

        # Give the server up to 2s to exit cleanly after stdin EOF.
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=2.0)
        except TimeoutError:
            self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(loop.run_in_executor(None, self._proc.wait), timeout=2.0)
            except TimeoutError:
                self._proc.kill()
                await loop.run_in_executor(None, self._proc.wait)

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def stderr_text(self) -> str:
        """Read accumulated stderr (test assertions check this)."""
        if not self._stderr_path.exists():
            return ""
        return self._stderr_path.read_text(encoding="utf-8", errors="replace")

    def stdout_lines_so_far(self) -> int:
        """Count how many newline-terminated lines have been written to
        stdout so far. Used by tests asserting "exactly N lines on stdout"
        — we already consumed them via _read_line, but this returns the
        cumulative request-counter as a proxy (one response per request)."""
        return self._next_id - 1


def _resolve_recall_binary() -> str:
    """Find the ``recall`` console script that lives in the same venv as
    the test runner.

    Resolution path: ``Path(sys.executable).parent / "recall"``. Under
    uv, ``sys.executable`` is ``<venv>/bin/python3``, so its parent
    dir holds the console scripts that ``pyproject.toml``'s
    ``[project.scripts]`` block creates at install time
    (``recall = "recall.cli:main"``).

    Why not ``python -m recall.cli serve``? ``src/recall/cli.py`` has
    no ``if __name__ == "__main__": main()`` guard, so invoking it via
    ``-m`` imports the module but never calls ``app()``. The process
    starts, exits cleanly with rc=0, prints nothing — silent and
    confusing failure mode. Initial 3.11 implementation hit this
    exact wall before pivoting to the console script.

    Why won't we add ``__main__`` to ``cli.py``? The CLI is invoked
    via the installed console script — that's the intentional public
    entrypoint and matches how ``recall`` is shipped on PyPI / via
    ``uvx recall-mcp``. Adding ``__main__`` would create a second
    entrypoint surface that diverges from how real users invoke
    Recall, exactly the kind of "test-only path" that drifts and
    masks production bugs (cf. CLAUDE.md "imports passing ≠ behavior
    correct" lesson from 2.5). We test the actual public entrypoint.

    3.12 inheritance: the recorded-session replay tests use the same
    ``mcp_subprocess_session`` / ``mcp_subprocess_factory`` fixtures
    and inherit this constraint. If 3.12 ever needs a different
    entrypoint, surface it explicitly during planning — don't quietly
    work around it here.
    """
    bin_dir = Path(sys.executable).parent
    recall_bin = bin_dir / "recall"
    if not recall_bin.exists():
        raise RuntimeError(
            f"recall console script not found at {recall_bin}. Did `uv sync` install the package?"
        )
    return str(recall_bin)


@pytest.fixture
async def mcp_subprocess_session(
    tmp_path: Path,
) -> AsyncIterator[Any]:
    """Spawn ``recall serve``; yield a session helper.

    Default environment: clean ``HF_HOME`` is NOT set (production cache
    used). Tests that need cold-cache isolation pass an override via the
    factory variant ``mcp_subprocess_factory`` below.

    Default ``RECALL_DB_PATH`` points at a unique non-existent path under
    tmp_path so the server boots in no-index state and tests don't leak
    into the maintainer's ~/.recall/db.sqlite.

    Cleanup: on test exit, closes stdin → waits 2s → SIGTERM → SIGKILL.
    """
    stderr_path = tmp_path / "recall_serve.stderr.log"
    env = os.environ.copy()
    env["RECALL_DB_PATH"] = str(tmp_path / "test-default-empty.sqlite")
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [_resolve_recall_binary(), "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_path.open("wb"),
        env=env,
        bufsize=0,  # Unbuffered — we want byte-for-byte stdout
    )

    session = MCPSubprocessSession(proc, stderr_path)
    try:
        yield session
    finally:
        await session.close()


@pytest.fixture
def mcp_subprocess_factory(tmp_path: Path):
    """Factory variant — tests parameterize env / db_path / etc. then own
    cleanup explicitly. Used by tests that need isolated_hf_cache or a
    pre-populated fixture DB.

    Returns an async function. Caller awaits it to spawn, manages
    teardown via ``await session.close()`` in a try/finally.
    """
    spawned: list[MCPSubprocessSession] = []

    async def _spawn(
        *,
        env_overrides: dict[str, str] | None = None,
        db_path: Path | None = None,
        stderr_filename: str = "recall_serve.stderr.log",
    ) -> MCPSubprocessSession:
        stderr_path = tmp_path / stderr_filename
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if db_path is not None:
            env["RECALL_DB_PATH"] = str(db_path)
        else:
            env["RECALL_DB_PATH"] = str(tmp_path / f"db-{len(spawned)}.sqlite")
        if env_overrides:
            env.update(env_overrides)

        proc = subprocess.Popen(
            [_resolve_recall_binary(), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_path.open("wb"),
            env=env,
            bufsize=0,
        )
        session = MCPSubprocessSession(proc, stderr_path)
        spawned.append(session)
        return session

    yield _spawn

    # Teardown: close all spawned sessions.
    async def _cleanup() -> None:
        for s in spawned:
            await s.close()

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async test — tests are responsible for
            # closing their own sessions; this is a fallback.
            for s in spawned:
                if s._proc.returncode is None:
                    s._proc.kill()
        else:
            loop.run_until_complete(_cleanup())
    except RuntimeError:
        # No event loop available; do best-effort sync cleanup.
        for s in spawned:
            if s._proc.returncode is None:
                s._proc.kill()


# === Cold-cache fixtures ===


@pytest.fixture
def isolated_hf_cache(tmp_path: Path) -> dict[str, str]:
    """Return env vars that point HuggingFace's cache to an empty tmpdir.

    F2 + §7 lock: cache-warming defeats the cold-cache test's purpose.
    Setting HF_HOME (and the legacy TRANSFORMERS_CACHE / HF_HUB_CACHE)
    to an empty tmpdir forces a fresh download-or-cache-load on every
    test invocation. The cost (~25s on Linux CPU) is the cost of
    correctness.

    Sanity check (per the verification refinement): cold-cache test
    runtime under 2s on M-series indicates the isolation isn't working
    — manual_smoke.md Section D documents this assertion.

    Network dependency: this fixture requires HF infra to serve the
    bge-small-en-v1.5 model. Failure mode if HF is down: clean
    network error in stderr, NOT a stdio-cleanliness false positive.
    """
    cache_root = tmp_path / "hf_cache"
    cache_root.mkdir()
    return {
        "HF_HOME": str(cache_root),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "HF_HUB_CACHE": str(cache_root / "hub"),
    }


@pytest.fixture
def fixture_indexed_db(tmp_path: Path) -> Path:
    """Build a tiny synthetic-vector DB matching the production schema.

    §7 option C lock: hand-insert 10 commands + their dim=384 vectors
    via the test_indexer.py FakeEmbedder384 pattern. Cheap (~10ms). The
    cold-cache subprocess test then dispatches search against this DB;
    the SUBPROCESS pays the embedder lazy-load cost when it encodes the
    query, which is the point of the cold-cache test.

    Returns the path to a migrated, populated DB at tmp_path/db.sqlite.
    """
    # Lazy import: keep conftest.py's import-time cost low for tests
    # that don't need this fixture.
    from recall.db import connect, migrate, set_meta
    from recall.embed import DEFAULT_MODEL

    db_path = tmp_path / "fixture-db.sqlite"
    conn = connect(db_path)
    migrate(conn)
    set_meta(conn, "embedding_model_name", DEFAULT_MODEL)

    # 10 plausible commands; deterministic hash for the test_hash; ts
    # spread over a few minutes so ORDER BY ts DESC is meaningful.
    commands = [
        ("zsh", "git status", 1000, "/home/u/proj"),
        ("zsh", "git diff", 1010, "/home/u/proj"),
        ("zsh", "ls -la", 1020, "/home/u/proj"),
        ("zsh", "cd /tmp", 1030, "/tmp"),
        ("zsh", "docker ps", 1040, "/home/u/proj"),
        ("zsh", "kubectl get pods", 1050, "/home/u/proj"),
        ("zsh", "psql -h localhost", 1060, "/home/u/proj"),
        ("zsh", "make test", 1070, "/home/u/proj"),
        ("zsh", "uv sync", 1080, "/home/u/proj"),
        ("zsh", "pytest -xvs", 1090, "/home/u/proj"),
    ]
    for source, text, ts, cwd in commands:
        # Synthetic dim=384 vector via the test_indexer pattern.
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        small = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
        small -= 128.0
        tiled = np.tile(small, 24).astype(np.float32)
        norm = float(np.linalg.norm(tiled))
        vec = tiled / (norm if norm > 0 else 1.0)

        text_hash = (text.encode("utf-8") + b"\x00" * 32)[:32]
        cur = conn.execute(
            "INSERT INTO commands (source, text_scrubbed, text_hash, cwd, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (source, text, text_hash, cwd, ts),
        )
        cmd_id = int(cur.lastrowid or 0)
        vec_blob = np.ascontiguousarray(vec.astype(np.float32)).tobytes()
        conn.execute(
            "INSERT INTO commands_vec (command_id, embedding) VALUES (?, ?)",
            (cmd_id, vec_blob),
        )
    conn.close()
    return db_path


# Expose sqlite3 so tests can construct connections via the standard module
# path even if the import doesn't appear used at top level.
_ = sqlite3
