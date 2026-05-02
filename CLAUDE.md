# Recall — guidance for Claude Code

Read this before doing any work in this repo. It captures decisions that the
code alone won't tell you.

## What Recall is

A locally-running MCP (Model Context Protocol) server that gives Claude Desktop
and Claude Code semantic, context-aware access to local shell history (zsh,
bash, fish, atuin). Public open-source project, MIT-licensed, targeted at a
Hacker News launch.

Distribution: PyPI as `recall-mcp`, runnable via `uvx recall-mcp`. Single-file
SQLite + `sqlite-vec` store at `~/.recall/db.sqlite`. Embedding model
`bge-small-en-v1.5` (~120 MB, dim 384) cached at `~/.recall/models/`. CPU-only.

Differentiation vs existing shell-history MCPs:

- Semantic search (vs substring elsewhere)
- First-class use of atuin metadata (cwd, exit_code, duration, hostname, session_id)
- Aggressive secret scrubbing as a documented trust feature
- Published `nl2bash` recall@k benchmark numbers in the README

## Critical correctness requirements

These are non-negotiable. Each fails CI.

### 1. Secret scrubber must not leak (`src/recall/scrub.py`)

The trust foundation of the project. Scrubs:

- AWS access keys (`AKIA[0-9A-Z]{16}`) and secret-key assignments
- GitHub tokens: `ghp_`, `gho_`, `ghs_`, `ghu_`, `github_pat_…`
- Generic bearer tokens, JWTs (3-part `eyJ…`)
- Password flags / env: `--password=`, `-p`, `PGPASSWORD=`, `MYSQL_PWD=`
- OpenAI / Anthropic keys (`sk-…`, `sk-ant-…`)
- Slack tokens (`xox[baprs]-`)
- Google API keys (`AIzaSy…`)
- Inline SSH private keys (`-----BEGIN … PRIVATE KEY-----`)
- URLs with `user:password@` userinfo
- **Flag-scoped** high-entropy values (Shannon > threshold) on these flags
  ONLY — never over free text:
  `--password`, `--token`, `--api-key`, `--secret`, `--access-key`, `--auth`,
  `Authorization: Bearer`, `?key=`, `?token=`, `?api_key=`,
  `X-API-Key:`, `X-Auth-Token:`

Replacement form: `<REDACTED:KIND>`. Idempotent: `scrub(scrub(s)) == scrub(s)`.

**Defense in depth: scrub at index time AND at query response.** Index-time
scrubbing means the on-disk SQLite never holds plaintext secrets. Dedup uses
`BLAKE2b(salt || raw)`; the salt is generated on first init and stored in the
`meta` table.

`pytest -k scrub` is the pre-commit canary. CI fails any PR that touches
`src/recall/scrub.py` or `tests/test_scrub.py` if the scrub test count
decreases (see `scrub-canary` job in `.github/workflows/ci.yml`).

### 2. zsh extended_history format

Handle `: <ts>:<dur>;<command>`, plain (no leading `:`) format, multi-line
continuations (trailing `\`), and invalid UTF-8 (latin-1 fallback). Never
crash on a malformed line — log and skip.

### 3. atuin schema robustness

Open the user's atuin database with `?mode=ro&immutable=1` so we never write
to it. Detect schema by checking for required columns at runtime, not by
hardcoded column order. Always `SELECT` by name.

### 4. Embedding consistency

Store embedding model name + revision in the `meta` table on first index.
On startup, refuse queries if the configured model differs from the indexed
one — prompt the user to run `recall index --rebuild`. Mixing embeddings
across models silently produces garbage results.

### 5. MCP protocol cleanliness

- Use the official `mcp` Python SDK, never hand-rolled.
- Tool input schemas validated via Pydantic. Bad input returns a clean MCP
  error, not a stack trace.
- Long operations (initial index) never run during tool calls. The server
  refuses to serve queries until the index exists; the CLI does the indexing.
- **stdio transport must not write anything to stdout except MCP frames.**
  All logging goes to stderr or `~/.recall/logs/recall.log`. A single stray
  `print()` silently breaks the server. There is a CI test (Phase 3) that
  spawns the server, sends `initialize`, and asserts every line on stdout is
  a valid MCP JSON-RPC frame.

### 6. Performance budgets (CI-enforced, Phase 3+)

- Cold start (model load + db open): < 2 s on M-series Mac
- Single semantic query over 50k commands: < 100 ms p95
- Initial index of 50k commands: < 60 s

Tests live at `tests/test_perf.py`.

### 7. Eval harness must run

`recall eval --dataset nl2bash` builds a fresh in-memory index from nl2bash
commands, runs all NL queries, reports recall@1, recall@5, MRR. Target:
recall@5 > 0.75 vs substring's ~0.2. These numbers ship in the README.

## MCP tool surface (locked signatures)

```text
search(query, limit=10, cwd_prefix=None, host=None, since=None) -> list[CommandHit]
find_in_project(query, limit=10, cwd=None) -> list[CommandHit]
commands_after(pattern, limit=10) -> list[SequenceHit]   # pattern is substring; no regex flag
failed_recently(window="24h", pattern=None, limit=20) -> list[CommandHit]
command_stats(pattern) -> CommandStats
recent(limit=20, cwd_prefix=None, host=None) -> list[CommandHit]
```

`find_in_project` cwd resolution order: explicit `cwd` arg > `MCP_CLIENT_CWD`
env > server-startup cwd. `since` accepts ISO date or relative ("7d", "2h",
"30m"). `failed_recently` requires the atuin source; returns a clear error
otherwise. Hybrid retrieval merges vector + FTS5 lexical via Reciprocal Rank
Fusion (k=60); falls back to vector-only when FTS5 unavailable.

Each tool returns scrubbed text only. Raw text never crosses the MCP boundary.

## Build order — strict phase gates

Stop at the end of each commit/phase and show progress before continuing.

**Phase 1 — foundations**
- 1.1 Project skeleton + tooling (this commit)
- 1.2 Scrubber + tests + secrets corpus fixture
- 1.3 DB schema, migrations, sqlite-vec wiring
- 1.4 Source readers: zsh, bash, atuin (fish deferred)

**Phase 2 — core retrieval**
- 2.5 Embedding wrapper, model caching, batch encoding
- 2.6 Hybrid search (vector + lexical, RRF k=60)
- 2.7 `recall index` CLI (full + incremental)
- 2.8 `recall eval` against nl2bash; report numbers

**Phase 3 — MCP surface**
- 3.9  `server.py` + `tools.py` with the six tools
- 3.10 stdout-cleanliness test
- 3.11 End-to-end MCP test (spawn, initialize, tool calls, assertions)

**Phase 4 — polish**
- 4.12 README with GIF, install, demo, benchmark table
- 4.13 PyPI publish workflow
- 4.14 Sample fixtures committed

## Commands

```bash
uv sync                          # install deps
uv run pytest                    # all tests
uv run pytest -k scrub           # scrubber canary (run before any commit
                                 #   that touches scrub.py)
uv run pytest -m "not embed"     # skip embedding-dependent tests
uv run ruff check .              # lint
uv run ruff format .             # apply formatting
uv run ruff format --check .     # CI: format check only
uv run mypy src                  # strict type check
```

## Workflow expectations

- **Plan-first.** On any non-trivial change, propose a plan and wait for
  approval before writing code.
- **Phase gates.** Stop at each commit boundary listed above and show the
  diff before continuing.
- Small commits with clear messages.
- Run tests after every meaningful change.
- Don't add dependencies until the commit that first uses them.
- Never `print()` to stdout in code paths that may run under the stdio MCP
  transport. Use `logging` (stderr or file).
- Never write to the user's atuin DB. Always open `?mode=ro&immutable=1`.
- Never commit raw history fixtures with real secrets. Synthesize them.
