"""Stdio cleanliness test suite (Commit 3.11).

Binds the CLAUDE.md §5 invariant: "after ``stdio_server()`` opens the
protocol loop, NO byte may reach stdout except an MCP JSON-RPC frame."

Six categories:
  Subprocess A — initialize-only round-trip
  Subprocess C — initialize + tools/call recent (no-index dispatch isolation)
  Subprocess D — cold-cache tools/call search (the marquee test)
  Subprocess E — lazy-refresh transition end-to-end (subprocess wiring)
  Static AST  — rejects sys.stdout writes / StreamHandler(sys.stdout) /
                print(..., file=sys.stdout) in src/recall/
  Static T201 — runs ruff --select T201 over src/recall/ as a subprocess

Lane: dedicated ``stdio`` lane on Ubuntu py3.12. Per F2 lock: Linux-only
for v1; M-series cleanliness verified via manual smoke (see
tests/manual_smoke.md Section D).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# === Subprocess test A: initialize-only round-trip ===


@pytest.mark.asyncio
async def test_initialize_only_clean_stdout(mcp_subprocess_session) -> None:
    """Send only ``initialize``; assert one valid MCP response on stdout
    and no Python tracebacks on stderr.

    Tests transport-level cleanliness independent of tool dispatch.
    """
    result = await mcp_subprocess_session.initialize()

    # JSON-RPC envelope shape
    assert result.get("jsonrpc") == "2.0"
    assert result.get("id") == 1
    assert "result" in result, f"expected result, got {result}"
    assert "error" not in result

    inner = result["result"]
    assert inner.get("protocolVersion"), "missing protocolVersion"
    assert inner.get("serverInfo", {}).get("name") == "recall-mcp"
    assert inner.get("serverInfo", {}).get("version") == "0.0.1"

    # Stderr cleanliness assertion (§8 strengthening): no tracebacks,
    # no WARNING / ERROR. INFO is fine.
    _assert_stderr_clean(mcp_subprocess_session.stderr_text())


# === Subprocess test C: tools/call recent (no-index dispatch isolation) ===


@pytest.mark.asyncio
async def test_recent_no_index_clean_stdout(mcp_subprocess_session) -> None:
    """Send initialize + initialized + tools/call recent against an empty
    DB path. Asserts the no-index error payload is well-formed.

    Diagnostic isolation: if this passes and D fails, the encode path is
    the culprit; if both fail, dispatch is.
    """
    init = await mcp_subprocess_session.initialize()
    assert init.get("result"), f"initialize failed: {init}"

    await mcp_subprocess_session.notify_initialized()

    resp = await mcp_subprocess_session.call_tool("recent", {"limit": 3})

    # JSON-RPC envelope
    assert resp.get("jsonrpc") == "2.0"
    assert "result" in resp
    inner = resp["result"]

    # State error: isError=True with the user-facing no_index message
    assert inner.get("isError") is True
    assert inner.get("structuredContent", {}).get("error"), (
        f"missing error in structuredContent: {inner}"
    )
    msg = inner["structuredContent"]["error"]
    assert "Index not found" in msg, f"unexpected error message: {msg}"

    _assert_stderr_clean(mcp_subprocess_session.stderr_text())


# === Subprocess test E: lazy-refresh transition end-to-end ===


@pytest.mark.asyncio
async def test_lazy_refresh_subprocess(
    mcp_subprocess_factory,
    tmp_path: Path,
) -> None:
    """Boot without a DB; first tools/call returns no_index; create the
    DB mid-session via direct INSERT (refinement: synthetic fixture, NOT
    a subprocess ``recall index`` call — that composition test belongs
    in 3.12); second tools/call returns results.

    Asserts the refresh transition fires end-to-end through the
    subprocess + MCP framing layers, not just the unit-tested function
    path in test_server.py.
    """
    db_path = tmp_path / "refresh-db.sqlite"
    session = await mcp_subprocess_factory(db_path=db_path)

    try:
        await session.initialize()
        await session.notify_initialized()

        # First call: no_index error.
        first = await session.call_tool("recent", {"limit": 5})
        assert first["result"]["isError"] is True
        assert "Index not found" in first["result"]["structuredContent"]["error"]

        # Build a tiny DB at the path the server is watching. Synthetic
        # fixture per the locked refinement — surgical test of "server
        # detects new DB and refreshes wiring," not of indexer integration.
        _build_tiny_indexed_db(db_path)

        # Second call: refresh fires; results returned. The protocol-
        # level transition (no_index → real results in a single process)
        # is the proof the refresh wiring is intact. The "state refreshed
        # — index detected" log line goes to recall.log (RotatingFileHandler
        # at ~/.recall/logs/), NOT stderr — setup_logging removes all
        # stderr handlers. We don't sniff the log file here; the unit
        # tests in test_server.py already cover the function path with
        # the log-line probe.
        second = await session.call_tool("recent", {"limit": 5})
        assert second["result"]["isError"] is False, f"refresh did not transition; got: {second}"
        results = second["result"]["structuredContent"]["results"]
        assert len(results) >= 1, "no results returned after refresh"

        _assert_stderr_clean(session.stderr_text())
    finally:
        await session.close()


# === Subprocess test D: cold-cache search (the marquee test) ===


@pytest.mark.asyncio
@pytest.mark.embed  # depends on sentence-transformers; runs in stdio lane
async def test_cold_cache_search_clean_stdout(
    mcp_subprocess_factory,
    isolated_hf_cache: dict[str, str],
    fixture_indexed_db: Path,
) -> None:
    """Marquee cold-cache test. Spawn ``recall serve`` with empty
    HF_HOME (forces fresh sentence-transformers download/load), send
    initialize + initialized + tools/call search.

    Assert: response is a well-formed JSON-RPC frame; no library
    progress bars / warnings / load messages on stdout. Asserts the
    redirect_stdout(sys.stderr) defense actually works under cold-load.

    Runtime: ~25s on Linux CPU. 90s timeout per the locked refinement
    R1 — distinguishes "HF download exceeded budget" from "stdout
    pollution" in CI logs.

    NETWORK DEPENDENCY: HF infra availability. If HF is down, this
    fails with a clear network error in stderr, NOT a stdio-cleanliness
    false positive. Recovery: if cold-cache fails on daily run with
    network error, re-run once. Twice consecutive = real signal (check
    status.huggingface.co or treat as regression).
    """
    session = await mcp_subprocess_factory(
        env_overrides=isolated_hf_cache,
        db_path=fixture_indexed_db,
    )
    try:
        await session.initialize()
        await session.notify_initialized()

        # 90s timeout — cold cache (fresh HF download + first encode +
        # MPS warmup, though MPS doesn't apply on Linux) is ~25s on
        # Linux CPU. The remaining headroom absorbs CI variance.
        try:
            resp = await session.call_tool(
                "search",
                {"query": "git status", "limit": 3},
                timeout=90.0,
            )
        except TimeoutError as e:
            stderr = session.stderr_text()
            pytest.fail(
                "cold-cache search exceeded 90s timeout. This is the "
                "DIAGNOSABLE-FAILURE marker (R1): could be HF download "
                "exceeded budget OR genuine stdout pollution masking the "
                "response. Inspect stderr below to distinguish.\n"
                f"original error: {e}\n"
                f"stderr (last 4000 chars):\n{stderr[-4000:]}"
            )

        # Response shape: well-formed JSON-RPC, well-formed CallToolResult
        assert resp.get("jsonrpc") == "2.0"
        assert "result" in resp, f"expected result, got: {resp}"
        inner = resp["result"]
        assert inner.get("isError") is False, (
            f"search returned error under cold cache:\n"
            f"  result: {inner}\n"
            f"  stderr: {session.stderr_text()[-2000:]}"
        )

        results = inner["structuredContent"]["results"]
        assert len(results) >= 1, "expected ≥1 search result against fixture DB"
        # Each result must have the expected CommandHit shape — cold cache
        # producing structurally-broken output would indicate the encode
        # path silently degraded.
        for r in results:
            assert "id" in r and "text" in r and "score" in r, (
                f"malformed CommandHit under cold cache: {r}"
            )

        # The load-bearing assertion: no library leaks on the stdout
        # path. We've already received valid JSON for every frame; the
        # check below ensures stderr (where progress bars SHOULD go,
        # if anywhere) doesn't contain Python tracebacks. Mere progress
        # bar text on stderr is fine.
        _assert_stderr_clean(session.stderr_text())
    finally:
        await session.close()


# === Static analysis: AST check ===


def test_no_stdout_writes_in_src() -> None:
    """Walk every .py under src/recall/; reject any direct stdout write.

    Rejection rules (§4 option B + flag refinement F4):
      - sys.stdout.write(...)
      - os.write(1, ...)         # fd 1 is stdout
      - StreamHandler(sys.stdout) / StreamHandler(stream=sys.stdout)
      - print(..., file=sys.stdout)

    The runtime gate `_assert_no_stdout_handler` covers the
    StreamHandler(sys.stdout) case at boot. This AST check makes the
    rule fail at PR time, before runtime, on code paths the tests don't
    execute.
    """
    src_root = Path(__file__).parent.parent / "src" / "recall"
    violations: list[str] = []

    for py_file in src_root.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError as e:
            pytest.fail(f"syntax error in {py_file}: {e}")

        for node in ast.walk(tree):
            v = _check_node(node, py_file, src_root)
            if v:
                violations.append(v)

    if violations:
        msg = "\n".join(violations)
        pytest.fail(
            f"AST check found {len(violations)} stdout-write violations:\n{msg}\n\n"
            "These bypass the redirect_stdout defense and break MCP framing. "
            "Use stderr or a file handler instead."
        )


def _check_node(node: ast.AST, py_file: Path, src_root: Path) -> str | None:
    """Return a violation string if this AST node writes to stdout, else None."""
    rel = py_file.relative_to(src_root.parent.parent)
    loc = f"  {rel}:{getattr(node, 'lineno', '?')}"

    # 1. sys.stdout.write(...)
    if isinstance(node, ast.Call):
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "write"
            and isinstance(f.value, ast.Attribute)
            and f.value.attr == "stdout"
            and isinstance(f.value.value, ast.Name)
            and f.value.value.id == "sys"
        ):
            return f"{loc}: sys.stdout.write(...) — write to stderr or use logging"

        # 2. os.write(1, ...) where 1 is fd stdout
        if (
            isinstance(f, ast.Attribute)
            and f.attr == "write"
            and isinstance(f.value, ast.Name)
            and f.value.id == "os"
            and len(node.args) >= 1
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 1
        ):
            return f"{loc}: os.write(1, ...) — fd 1 is stdout; use stderr (fd 2)"

        # 3. StreamHandler(sys.stdout) — positional or keyword form
        if (isinstance(f, ast.Name) and f.id == "StreamHandler") or (
            isinstance(f, ast.Attribute) and f.attr == "StreamHandler"
        ):
            for arg in node.args:
                if _is_sys_stdout(arg):
                    return f"{loc}: StreamHandler(sys.stdout) — must target stderr or a file"
            for kw in node.keywords:
                if kw.arg == "stream" and _is_sys_stdout(kw.value):
                    return f"{loc}: StreamHandler(stream=sys.stdout) — must target stderr or a file"

        # 4. print(..., file=sys.stdout) — closes the gap T201 might miss
        if isinstance(f, ast.Name) and f.id == "print":
            for kw in node.keywords:
                if kw.arg == "file" and _is_sys_stdout(kw.value):
                    return f"{loc}: print(..., file=sys.stdout) — bypass via file= kwarg"

    return None


def _is_sys_stdout(node: ast.AST) -> bool:
    """True if ``node`` is an AST reference to ``sys.stdout``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "stdout"
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    )


# === Static analysis: T201 invocation ===


def test_ruff_t201_clean() -> None:
    """Run ``ruff check --select T201 src/recall/`` directly. Asserts
    zero T201 violations.

    Belt-and-suspenders against someone bypassing the global ruff config
    (e.g. a per-file ignore that accidentally widens scope). Independent
    of the per-PR ``ruff check .`` step.
    """
    src_root = Path(__file__).parent.parent / "src" / "recall"
    result = subprocess.run(
        ["ruff", "check", "--select", "T201", str(src_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"ruff T201 violations in src/recall/:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


# === Helpers ===


def _assert_stderr_clean(stderr: str) -> None:
    """Assert stderr contains no Python tracebacks and no WARNING/ERROR
    log lines (§8 strengthening). INFO is fine.

    Filters out the uv/setup environment warnings that aren't from
    recall.* loggers — those are infrastructure, not server cleanliness.
    """
    if "Traceback" in stderr:
        pytest.fail(f"Python traceback in stderr:\n{stderr[-4000:]}")

    # Walk lines; flag any line containing WARNING or ERROR that's a
    # recall.* log entry. We match the ``setup_logging`` formatter:
    #   "%(asctime)s %(levelname)s %(name)s %(message)s"
    # so WARNING / ERROR appear in column 2.
    bad: list[str] = []
    for line in stderr.splitlines():
        # Skip uv setup chatter and the venv mismatch warning that
        # appears even in passing runs.
        if "VIRTUAL_ENV" in line or "Building recall" in line or "Built recall" in line:
            continue
        if "Uninstalled" in line or "Installed" in line:
            continue
        # Match recall logger format: "<ts> WARNING <name> ..."
        for level in ("WARNING", "ERROR", "CRITICAL"):
            # Look for level token surrounded by whitespace (not WARN-ed
            # or NotEr...; clear word boundary).
            if f" {level} " in f" {line} ":
                bad.append(line)
                break
    if bad:
        pytest.fail(
            f"stderr contains {len(bad)} WARNING/ERROR/CRITICAL line(s):\n" + "\n".join(bad)
        )


def _build_tiny_indexed_db(db_path: Path) -> None:
    """Hand-build a small DB at db_path matching the production schema.

    Used by test E (lazy-refresh subprocess). Synthetic vectors per the
    locked refinement — surgical "did the server detect a new DB?"
    test, not an indexer integration test.
    """
    import hashlib

    from recall.db import connect, migrate, set_meta
    from recall.embed import DEFAULT_MODEL

    conn = connect(db_path)
    migrate(conn)
    set_meta(conn, "embedding_model_name", DEFAULT_MODEL)

    text = "test command for refresh"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    small = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
    small -= 128.0
    tiled = np.tile(small, 24).astype(np.float32)
    norm = float(np.linalg.norm(tiled))
    vec = tiled / (norm if norm > 0 else 1.0)
    text_hash = (text.encode("utf-8") + b"\x00" * 32)[:32]
    cur = conn.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", text, text_hash, 1000),
    )
    cmd_id = int(cur.lastrowid or 0)
    vec_blob = np.ascontiguousarray(vec.astype(np.float32)).tobytes()
    conn.execute(
        "INSERT INTO commands_vec (command_id, embedding) VALUES (?, ?)",
        (cmd_id, vec_blob),
    )
    conn.close()


# Used by sys-stdout reference. Quiet linters that flag the import as
# unused (it's referenced via ast in the AST check, not at module level).
_ = sys
