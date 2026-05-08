# Manual smoke procedure — MCP server (3.9 + 3.10)

This is the only thing exercising actual stdio-transport before
3.11 lands the CI subprocess test. It's where SDK-runner integration
bugs and stdout-pollution leaks would surface (the kind the in-process
tests in `test_server.py` and `test_tools.py` can't reach).

Run this procedure manually after every meaningful change to
`src/recall/server.py`, `src/recall/tools.py`, `src/recall/cli.py::serve_cmd`,
or `pyproject.toml` (mcp dep).

The procedure has three sections:
  - **Section A: 3.9-era initialize handshake** (was the whole smoke at 3.9).
  - **Section B: 3.10 tool-call exercise** — one JSON-RPC `tools/call`
    per tool, asserting shape and stdout cleanliness. Requires an
    indexed DB at `~/.recall/db.sqlite` (build via `recall index`).
  - **Section C: live MCP-client smoke (Claude Desktop)** — the
    closest 3.10 will come to real-client testing before 3.13.
    Configure `verify/3.10` server, confirm `tools/list` renders all
    six with descriptions, run at least one `tools/call` and confirm
    the response renders. Surface anything off-looking before merge.

## Prerequisites

1. `uv sync` succeeded; `recall serve` is on PATH inside the venv
2. Stdout/stderr are TTY-attached (i.e. you're running it from a real
   shell, not piped to a non-terminal that buffers differently)
3. For Section A (initialize): `~/.recall/db.sqlite` may or may not
   exist; both states are valid (the no-index branch is covered by
   in-process tests).
4. For Section B (tool-call exercise): `~/.recall/db.sqlite` must
   exist with at least a few hundred rows. Build via `recall index`
   if needed.

<!--
3.12 cross-reference: the recorded MCP protocol scenarios in
tests/test_recorded_sessions.py (replaying tests/fixtures/sessions/*.jsonl)
are the CI-enforced equivalent of Sections A and B. The manual procedure
below stays as the maintainer's local M-series gate; the replay tests
catch the same regressions on every PR via the stdio CI lane.
-->

## Section A — 3.9-era initialize handshake

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

## Section A: Last verified output

Verified during 3.9 implementation (M-series, fresh `~/.recall/db.sqlite`
absent — exercising the no-index startup path).

```
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"experimental":{},"tools":{"listChanged":false}},"serverInfo":{"name":"recall-mcp","version":"0.0.1"}}}
```

Stdout line count: 1. Stderr: empty. Process exit: clean (~2s including
`uv run` startup overhead).

---

## Section B — 3.10 tool-call exercise

The 3.10 commit ships six tools. This section confirms each can be
called over stdio without polluting stdout. The same prerequisites
hold (TTY shells, repo-root invocation); additionally the local
`~/.recall/db.sqlite` should be a real index (run `recall index`
first).

The general shape: pipe `initialize` + `initialized` notification +
`tools/list` + one `tools/call` per tool to `recall serve`, capturing
stdout. The MCP SDK requires the `initialized` notification to be
sent before tool calls.

### Helper: pipe a sequence of frames

Save the following one-shot script into a scratch file (e.g.
`/tmp/recall_call.sh`) and `chmod +x` it; the rest of this section
calls into it.

```bash
#!/usr/bin/env bash
# Usage: /tmp/recall_call.sh '<tools/call params JSON>'
# Example: /tmp/recall_call.sh '{"name":"recent","arguments":{"limit":3}}'
#
# Note: the `sleep 0.2` between frames is NOT cosmetic. When piping a
# static script into `recall serve`, stdin EOFs immediately after the
# last echo; the SDK's stdio reader can finish processing and tear down
# the lifecycle before queued frames are fully drained, dropping the
# last response. Real MCP clients (Claude Desktop, Cursor, etc.) hold
# stdin open across the whole session and don't hit this. The delays
# are static-pipe-test scaffolding only.
set -euo pipefail
PARAMS="$1"
{
  echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual-smoke","version":"0.1.0"}}}'
  sleep 0.2
  echo '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
  sleep 0.2
  echo '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'
  sleep 0.2
  echo "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"tools/call\",\"params\":${PARAMS}}"
  sleep 0.5
} | uv run recall serve 2>/tmp/recall_serve.stderr.log
```

### Common assertions for every tool call

For each `tools/call` you run below:

- [ ] Stdout contains exactly **3 lines** (initialize response,
      tools/list response, tools/call response). No fourth line —
      that would indicate stdout pollution somewhere in the encode
      path (the `redirect_stdout(sys.stderr)` defense should prevent
      this even if a library tries to print). If you see only 2
      lines, increase the `sleep` durations in the helper script —
      stdin EOFed before the last frame was drained.
- [ ] All three lines parse as valid JSON.
- [ ] The `tools/call` response (id: 3) has `result.content[0].type ==
      "text"`.
- [ ] If `result.isError == true`, the text is the user-facing error
      message (e.g. "Index not found at..."); not a Python traceback.
- [ ] `/tmp/recall_serve.stderr.log` MUST NOT contain a Python
      traceback. Logging output (INFO-level "tool=… ok count=…") is
      fine and expected.

### Tool 1: `recent`

```bash
/tmp/recall_call.sh '{"name":"recent","arguments":{"limit":3}}'
```

Expected `result.structuredContent.results` is an array of up to 3
objects; each has `id`, `text`, `text_hash` (16-hex-char), `source`,
`ts`. `score` field will be `null` (recent isn't semantic).

### Tool 2: `search`

```bash
/tmp/recall_call.sh '{"name":"search","arguments":{"query":"git status","limit":3}}'
```

Expected: 3 results, each with `score` ∈ [-1, 1] (cosine similarity);
descending. The first call pays ~5s for the embedder lazy-load (you'll
see this in `/tmp/recall_serve.stderr.log` as
`recall.server: lazy-loading embedder (first tool call)`).

### Tool 3: `find_in_project`

```bash
/tmp/recall_call.sh '{"name":"find_in_project","arguments":{"query":"build","cwd":"/tmp"}}'
```

Expected: results scoped to commands run under `/tmp` (likely
empty or very few — that's fine; the test is that the response is
shaped correctly, not that the results are interesting).

If `cwd` is omitted, `find_in_project` falls back to `MCP_CLIENT_CWD`
env or the server's startup cwd. Run from the repo root if you want
to see commands in this project's context.

### Tool 4: `commands_after`

```bash
/tmp/recall_call.sh '{"name":"commands_after","arguments":{"pattern":"git","limit":2}}'
```

Expected: each result has `pattern_match` (a CommandHit) and
`following` (an array of CommandHits run in the same session
afterwards; up to 3). Use `git` as the pattern because almost any
shell history will have several matches.

### Tool 5: `failed_recently`

```bash
/tmp/recall_call.sh '{"name":"failed_recently","arguments":{"window":"7d"}}'
```

Expected if you have atuin indexed: an array of CommandHits with
non-zero `exit_code`. Expected if you only have zsh/bash indexed
(no atuin): `result.isError == true` with the message
`"failed_recently requires the 'atuin' source..."`. That's the
correct outcome for source-presence-failure — verify the message
renders cleanly, not as a stack trace.

### Tool 6: `command_stats`

```bash
/tmp/recall_call.sh '{"name":"command_stats","arguments":{"pattern":"docker"}}'
```

Expected: `result.structuredContent` has `top_cwds` (list of `[cwd,
count]` pairs), `mean_duration_ms`, `success_rate`, `by_source`,
`total`. JSON-serialized as nested arrays/objects.

---

## Section C — Live MCP-client smoke (Claude Desktop)

This is the closest 3.10 will come to real-MCP-client testing
before 3.13. Required step before squash-merge.

### Procedure

1. **Configure Claude Desktop** to use the `verify/3.10` build of
   `recall serve`. Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

   ```json
   {
     "mcpServers": {
       "recall": {
         "command": "/path/to/repo/.venv/bin/recall",
         "args": ["serve"]
       }
     }
   }
   ```

   Restart Claude Desktop.

2. **Confirm tools/list renders**: open a new conversation, click the
   tools indicator (hammer/wrench icon), and verify all six tools
   appear with their descriptions:
   - search
   - find_in_project
   - commands_after
   - failed_recently
   - command_stats
   - recent

3. **Run at least one tools/call**: ask Claude Desktop something
   like "use the recent tool to show me my last 5 commands" or
   "search my shell history for git commands". Confirm:
   - The response renders in the UI (no raw error blob)
   - Result text is scrubbed (no real secrets visible — should be
     `<REDACTED:...>` markers if your history had any)
   - No stack-traces visible to the user

4. **Check `~/.recall/logs/recall.log`** has structured INFO lines
   like `tool=recent ok count=5`. Query text NOT logged (Q8 policy).

### Assertion checklist

- [ ] All six tools listed in Claude Desktop's tool palette
- [ ] Tool descriptions render (truncated by client UI is fine, but
      no missing/empty descriptions)
- [ ] At least one `tools/call` runs end-to-end and renders
- [ ] No tracebacks visible in client UI
- [ ] No stdout-pollution (would manifest as broken JSON / dropped
      messages from the client's perspective)

### What to do if anything looks off

Surface the issue in the squash-merge PR comment **before merge**.
This is the last opportunity to catch a real-MCP-client UX regression
before the 3.13 documented-clients-tested checklist runs across
multiple clients.

---

## What to do if any assertion fails

(applies to all three sections)

- **Multiple stdout lines** → some library is writing to stdout.
  Common culprits: `huggingface_hub` HTTP request lines, `torch`
  MPS warnings, `sentence_transformers` "Loading weights" progress.
  3.10's `redirect_stdout(sys.stderr)` around `embedder.encode`
  should catch most of these. If a leak gets through, the source is
  outside the encode call (e.g. transitive import-time print) — fix
  the source.
- **Parse error / not-JSON** → server crashed during initialization
  or tool dispatch. The stderr log will have the traceback.
- **Server hangs after one response** → stdio_server is waiting for
  more input. Acceptable for a real MCP client; for these smokes we
  EOF stdin via end-of-pipe. If the server doesn't exit, that's a
  SDK API misuse — check the lifecycle handling in `main()`.
- **Wrong serverInfo** → `_SERVER_NAME` / `_SERVER_VERSION` constants
  in server.py drifted from the smoke's expected values.
- **Tool-call returns "Index not found"** → no DB. Run `recall index`
  to build one.

---

## Section B: Last verified output

Verified during 3.10 implementation (M-series, fresh
`~/.recall/db.sqlite` ABSENT — exercising the no-index error path
through the full tool dispatch).

### `recent` (no_index error path)

```
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"Index not found at /Users/<u>/.recall/db.sqlite. Run 'recall index' to build one from your shell history."}],"structuredContent":{"error":"Index not found at /Users/<u>/.recall/db.sqlite. Run 'recall index' to build one from your shell history.","results":[]},"isError":true}}
```

Stdout line count: 3 (initialize + tools/list + tools/call).
Stderr: empty. No tracebacks. Process exit: clean.

### `command_stats` with bare wildcard (validation error path)

```
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"pattern cannot be bare SQL wildcards (e.g. '%', '%%'); provide a literal substring."}],"structuredContent":{"error":"pattern cannot be bare SQL wildcards (e.g. '%', '%%'); provide a literal substring.","results":[]},"isError":true}}
```

Stdout line count: 2 (initialize + tools/call; no tools/list in this
run). Stderr: empty. No tracebacks. Process exit: clean.

The full Section B (each of the six tools against an indexed DB) +
Section C (Claude Desktop UX confirmation) are run by the maintainer
before squash-merge of this branch and again at every meaningful
change to `tools.py` or `server.py`.

---

## Section D — stdio-cleanliness suite (3.11)

This section is the in-process equivalent of the `stdio (ubuntu /
py3.12)` CI lane added in Commit 3.11 — same tests, run locally on
M-series. The CI lane is Linux-only per the F2 lock at 3.11 plan;
this section is **the maintainer's M-series local verification**,
which is a v1.0 launch-checklist item per CLAUDE.md "Deferred items."

The lane source lives at `tests/test_stdio_cleanliness.py`. Six tests:

- **A** initialize-only round-trip
- **C** initialize + tools/call recent (no-index dispatch isolation)
- **D** cold-cache tools/call search (the marquee test — empty
  HF_HOME, fresh model download)
- **E** lazy-refresh transition end-to-end (subprocess wiring of
  the 3.10 first-run UX fix)
- **AST check** rejects sys.stdout writes / StreamHandler(sys.stdout)
  / print(..., file=sys.stdout) in src/recall/
- **T201 invocation** runs `ruff check --select T201 src/recall/`
  as a subprocess

### Pre-run sanity check (the §7 isolation refinement)

Before running test D, verify two things:

1. **Warm HF cache exists** — `~/.cache/huggingface` should already
   hold the bge-small-en-v1.5 model from prior dev work. This is the
   BASELINE the test is contrasting against: when isolation works,
   the test pays a real download/load cost (~5-25s on M-series, ~25s
   on Linux). When isolation is broken, the test would unintentionally
   reuse this warm cache and complete in <2s.

   ```bash
   ls -d ~/.cache/huggingface 2>&1 || \
     echo "WARNING: no HF cache yet; first run will pay full download cost"
   ```

2. **Test D pays load cost** — assert wall time is ≥2s (M-series).
   If completion <2s, the `isolated_hf_cache` fixture isn't actually
   isolating — investigate before trusting any result.

   ```bash
   time uv run pytest tests/test_stdio_cleanliness.py::test_cold_cache_search_clean_stdout
   ```

### The procedure

```bash
# Non-embed tests (A, C, E, AST, T201) — fast (~5s)
uv run pytest tests/test_stdio_cleanliness.py -v -m "not embed"

# Cold-cache marquee test — separate so runtime is visible
time uv run pytest tests/test_stdio_cleanliness.py::test_cold_cache_search_clean_stdout -v
```

### Assertion checklist

- [ ] Tests A, C, E, AST check, T201 all pass (~5s combined).
- [ ] Test D passes; wall time ≥2s on M-series (confirms isolation).
- [ ] No Python tracebacks anywhere.
- [ ] No WARNING/ERROR/CRITICAL log lines (`_assert_stderr_clean`
      fires if any).

### Last verified output (M-series local, 3.11 implementation)

Tests A, C, E, AST check, T201 ran in **1.84s** combined. Test D
(cold-cache, with `isolated_hf_cache` forcing fresh model load via
empty HF_HOME) ran in **22.65s** — well above the 2s sanity
threshold, confirming cache isolation. All 6 tests passed; zero
tracebacks; zero stderr warnings.

```
tests/test_stdio_cleanliness.py::test_initialize_only_clean_stdout PASSED
tests/test_stdio_cleanliness.py::test_recent_no_index_clean_stdout PASSED
tests/test_stdio_cleanliness.py::test_lazy_refresh_subprocess PASSED
tests/test_stdio_cleanliness.py::test_no_stdout_writes_in_src PASSED
tests/test_stdio_cleanliness.py::test_ruff_t201_clean PASSED
tests/test_stdio_cleanliness.py::test_cold_cache_search_clean_stdout PASSED  [22.65s]
```

### Pre-launch (v1.0 announce) re-run

Per CLAUDE.md "Deferred items" → "v1.0 launch checklist: cold-cache
subprocess test on local M-series Mac" — re-run test D before v1.0
announcement and append the result here (or to `docs/clients-tested.md`
once Commit 3.13 lands that file).
