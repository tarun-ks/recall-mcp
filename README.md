# Recall — semantic shell history MCP

Recall is a locally-running MCP (Model Context Protocol) server that gives
Claude Desktop and Claude Code semantic, context-aware access to your shell
history.

**Status:** pre-alpha — Phase 1 complete (`v0.1.0-phase1`). Foundations
land in this tag: scrubber, storage, source readers. **No MCP server yet,
no indexer yet.** Not on PyPI; the README and install flow remain
intentionally minimal until Phase 4 polish.

## What's built so far (Phase 1)

- **Secret scrubber** (`recall.scrub`) — pattern pack + flag-scoped
  entropy detector covering AWS access/secret keys, GitHub tokens, JWTs
  and bearer tokens, OpenAI / Anthropic / Slack / Google API keys,
  inline SSH private keys, password flags, URL userinfo, and sensitive-
  flag values. Idempotent (`scrub(scrub(s)) == scrub(s)`). 119-test
  canary suite is the pre-commit gate. Designed to scrub at index time
  so the on-disk SQLite never holds plaintext secrets.
- **SQLite + `sqlite-vec` storage** (`recall.db`) — schema v1 with
  `commands`, `commands_vec` (vec0 `FLOAT[384]`), and `commands_fts`
  (FTS5 external-content). BLAKE2b dedup hashing with a salt preserved
  across rebuilds; `--new-salt` is an explicit escape hatch.
- **Source readers** (`recall.sources`) — `ZshSource` (extended + plain
  + multi-line continuations + latin-1 fallback), `BashSource` (with
  `HISTTIMEFORMAT` support), `AtuinSource` (read-only/immutable open,
  runtime schema introspection, no journal sidecars). Unified
  `HistorySource` protocol with locked `iter_entries(since)` semantics.

## What's not yet built

- **Indexer** (`recall index` CLI) — Phase 2. Until this lands, nothing
  populates the database; the storage layer is ready and waiting.
- **Embedding pipeline** — `bge-small-en-v1.5` integration, model
  caching, batch encoding — Phase 2.
- **Hybrid search** — vector + FTS5 with reciprocal rank fusion (k=60),
  vector-only fallback when FTS5 unavailable — Phase 2.
- **`recall eval`** against `nl2bash` — Phase 2; `recall@1` / `recall@5` /
  MRR numbers will ship in this README before the first tagged release.
- **MCP server** (`recall-mcp` stdio transport) with the six tools
  (`search`, `find_in_project`, `commands_after`, `failed_recently`,
  `command_stats`, `recent`) — Phase 3.
- **Secret-scrubber expanded pattern coverage** — Phase 4, tracked as a
  v1 launch blocker. The current 18-pattern set covers AWS, GitHub
  tokens, JWTs / bearer tokens, OpenAI / Anthropic, Slack, Google API
  keys, and generic password / token / API-key flags (including URL
  query parameters and HTTP headers like `X-API-Key:` and
  `X-Auth-Token:`). It does NOT yet cover Stripe (`sk_live_…`,
  `pk_live_…`), npm tokens (`npm_…`), Heroku / Cloudflare API tokens,
  Twilio account SIDs, or a few other ecosystem-specific formats.
- **Fish source reader** — Phase 4. The `HistorySource` protocol is
  ready; only the implementation is deferred.

## Planned features (full v1)

- Semantic search over zsh, bash, fish, and atuin history
- Project-scoped retrieval, sequence patterns, failure analysis
- First-class secret scrubbing — your secrets stay local and never reach
  the LLM
- Single-file SQLite + `sqlite-vec` store, ~120 MB embedding model on
  CPU, fully offline
- Published `nl2bash` `recall@k` benchmark numbers in this README

Install instructions, demo GIF, and benchmark numbers will land before
the first tagged v1 release.

## Contributing

This is pre-alpha; the API and CLI surface will change. If you're poking
around and want to run the test suite locally:

```bash
uv sync                          # recommended
# or: python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"

uv run pytest                    # full suite
uv run pytest -k scrub           # scrubber canary (the pre-commit gate)
uv run ruff check .
uv run mypy src
```

License: MIT.
