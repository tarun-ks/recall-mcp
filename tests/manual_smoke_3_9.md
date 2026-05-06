# Manual smoke procedure — Commit 3.9 (server skeleton)

This is the only thing exercising actual stdio-transport in 3.9. It's
where SDK-runner integration bugs would surface (the kind the in-process
tests in `test_server.py` can't reach). Not in CI yet — the CI subprocess
test lands in 3.11.

Run this procedure manually after every meaningful change to
`src/recall/server.py`, `src/recall/cli.py::serve_cmd`, or
`pyproject.toml` (mcp dep).

## Prerequisites

1. `uv sync` succeeded; `recall serve` is on PATH inside the venv
2. Stdout/stderr are TTY-attached (i.e. you're running it from a real
   shell, not piped to a non-terminal that buffers differently)
3. `~/.recall/db.sqlite` may or may not exist; both states are valid
   for this smoke (3.9's "no-index" branch was also covered by the
   in-process tests)

## The procedure

Run the following from the repo root:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-smoke","version":"0.0.1"}}}' \
  | uv run recall serve 2>/tmp/recall_serve.stderr.log
```

Capture EVERYTHING that comes back on stdout. `2>` redirects stderr
into a file so it doesn't pollute the visual output you'll inspect.

## Required response shape

Output should be **exactly one line of valid JSON**, the MCP
`initialize` response. The shape:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { ... },
    "serverInfo": {
      "name": "recall-mcp",
      "version": "0.0.1"
    }
  }
}
```

## Assertion checklist

Walk through each of these. Check them off in this file (or paste the
log entry) when verifying.

- [ ] **Stdout has exactly one line of output** (count via
      `... | wc -l` should be `1`). Multiple lines on stdout means
      something else is leaking — model-load progress, library
      warnings, anything. That's the launch-day risk this smoke
      catches.
- [ ] **The line parses as valid JSON.** Pipe to
      `python -m json.tool` and confirm no parse error.
- [ ] **`jsonrpc` field is `"2.0"`.**
- [ ] **`id` field matches the request id (`1`).**
- [ ] **`result` (not `error`) is present.** An `error` payload here
      means the server failed to initialize; stderr log will have
      the traceback.
- [ ] **`result.serverInfo.name == "recall-mcp"`** and
      **`result.serverInfo.version == "0.0.1"`**. These are pinned in
      `recall/server.py`'s `_SERVER_NAME` and `_SERVER_VERSION`
      constants; updates to those constants need this smoke to update too.
- [ ] **`result.protocolVersion`** is a non-empty string (the SDK
      version-negotiates).
- [ ] **`result.capabilities`** is an object with at least the
      `tools` key (since the server registered a `list_tools` handler
      even if the list is empty).
- [ ] **The process exits cleanly** after the single response — no
      hang waiting for more input (since stdin closed after our `echo`).
      Concretely: the shell prompt returns within ~5 seconds.
- [ ] **`/tmp/recall_serve.stderr.log` may have logging output** (that's
      fine — INFO logs go to file by default but warnings can land on
      stderr). What it must NOT contain: a Python traceback.

## What to do if any assertion fails

- **Multiple stdout lines** → some library is writing to stdout. Common
  culprits: `huggingface_hub` HTTP request lines, `torch` MPS warnings,
  `sentence_transformers` "Loading weights" progress. The fix is to
  silence the source (3.10's `redirect_stdout` context manager will be
  the systemic defense; for 3.9 we expect zero stdout pollution because
  no embedder loads here).
- **Parse error / not-JSON** → server crashed during initialization. The
  stderr log will have the traceback.
- **Server hangs after one response** → stdio_server is waiting for more
  input. Acceptable for an MCP client (it would send `initialized`
  notification + tool calls), but for this smoke we send just one frame
  and EOF stdin. If the server doesn't exit, that's a SDK API misuse —
  check that the lifecycle is correctly handling stdin EOF.
- **Wrong serverInfo** → `_SERVER_NAME` / `_SERVER_VERSION` constants
  in server.py drifted from the smoke's expected values.

## Capture for posterity

When this procedure passes cleanly, paste the actual stdout JSON into
this file (under "Last verified output" below) so future-you can
spot-diff against future runs.

## Last verified output

Verified during 3.9 implementation (M-series, fresh `~/.recall/db.sqlite`
absent — exercising the no-index startup path).

```
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"experimental":{},"tools":{"listChanged":false}},"serverInfo":{"name":"recall-mcp","version":"0.0.1"}}}
```

Stdout line count: 1. Stderr: empty. Process exit: clean (~2s including
`uv run` startup overhead).
