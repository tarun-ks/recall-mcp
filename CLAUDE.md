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

**Adding a new scrubber pattern — required additions:**
1. **Idempotency check.** Verify the new pattern doesn't match against
   `<REDACTED:*>` markers from any other pattern. Idempotency is by
   construction *only* if patterns are mutually exclusive on their
   replacements (cf. the `URL_USERINFO`-vs-`<REDACTED:GITHUB_TOKEN>` bug
   fixed in 1.2 where `<` `>` weren't excluded from the userinfo char
   classes). **Add at least one idempotency test per new pattern**, even
   if `test_idempotent` already runs over your input — explicit beats
   implicit when patterns interact.
2. **Corpus growth.** Add ≥ 1 positive AND ≥ 1 negative line to
   `tests/fixtures/secrets_corpus.txt` for the new pattern. The corpus
   integrity test (`test_corpus_no_known_leaks`) scales in value with
   corpus diversity — a per-test case that misses a leak still has to
   pass the corpus pass.

**Canary end-to-end verification status (as of Commit 1.2):** the workflow's
shell logic was manually validated locally — path filter, base/head count
differentiation, comparison, and working-tree restoration all confirmed.
But the job has never actually run on real GitHub Actions (the repo has no
remote yet — first push is intentionally deferred to the end of Phase 1).
**The 1.2 PR is the implicit end-to-end verification.** When 1.2's PR opens
against the freshly pushed remote, the `scrub-canary` job MUST appear in the
Actions tab and execute. If it doesn't appear at all, the path filter or
trigger is wrong and that is a 1.2 blocker — fix before merging anything.

### 2. zsh extended_history format

Handle `: <ts>:<dur>;<command>`, plain (no leading `:`) format, multi-line
continuations (trailing `\`), and invalid UTF-8 (latin-1 fallback). Never
crash on a malformed line — log and skip.

### 2a. HistorySource protocol semantics

All sources implement `HistorySource` (in `src/recall/sources/base.py`) and
expose exactly one method:

`iter_entries(since: int | None = None) -> Iterator[Entry]`

- `since` is **wall-clock unix seconds**. Wall-clock is the only definition
  that survives multi-source merging in the Phase 2 indexer (atuin's
  nanosecond timestamps, zsh's `EXTENDED_HISTORY` seconds, and bash's
  `HISTTIMEFORMAT` seconds all collapse to the same axis). Sources that
  store other units convert at the iterator boundary.
- Sources yield entries with `ts > since` in source-native order. Sources
  MAY also emit entries with `ts <= since` if they detect a cursor
  mismatch (histfile rewrite, file truncated and rebuilt, etc.) — that
  backfill-on-mismatch logic is encapsulated per source. The Phase 2
  indexer deduplicates on `UNIQUE(source, text_hash, ts)` regardless.
- Entries with an unknown timestamp emit `ts = 0` and are **always** yielded
  regardless of `since` (no way to know if they're old or new; the indexer's
  UNIQUE constraint catches duplicates).
- Sources are stateless across calls. Each `iter_entries` opens whatever
  file or DB it needs and closes when the iterator is exhausted.

### 3. atuin schema robustness

Open the user's atuin database with `?mode=ro&immutable=1` so we never write
to it. Detect schema by checking for required columns at runtime, not by
hardcoded column order. Always `SELECT` by name.

### 4. Embedding consistency

Store embedding model name + revision in the `meta` table on first index.
On startup, refuse queries if the configured model differs from the indexed
one — prompt the user to run `recall index --rebuild`. Mixing embeddings
across models silently produces garbage results.

### 4a. Dedup salt and rebuild policy

The dedup hash is `BLAKE2b(salt ‖ raw_text)`, output 32 bytes, stored as
`commands.text_hash`. The salt lives in `meta.dedup_salt` (hex), with
`meta.dedup_salt_version` tracking rotations.

**Lifecycle (all flags described here are spec for `recall index`, which
lands in Commit 2.7):**

| Command | Salt | Commands table |
| --- | --- | --- |
| `recall index` (incremental) | unchanged | append-only |
| `recall index --rebuild` | **unchanged** (default — conservative) | cleared, refilled |
| `recall index --rebuild --new-salt` | rotated, version++ | cleared, refilled |
| `recall index --new-salt` (no `--rebuild`) | **rejected with error** | n/a |

**Invariant:** every row's `text_hash` was computed with the salt current
at the time of insertion. Salt rotation MUST be paired with full
re-insertion of all rows (otherwise old-salt and new-salt hashes coexist
in one table, silently breaking dedup). The CLI rejects `--new-salt`
without `--rebuild` for this reason.

**Why the default preserves the salt across rebuilds:** identical raw
text hashes to the same value before and after a rebuild, so external
tooling and incremental syncs against the post-rebuild DB stay coherent.
`--new-salt` is an explicit escape hatch for users who suspect their salt
is corrupted; the CLI help text documents the tradeoff.

**API split:** `db.py` exposes `rotate_dedup_salt(conn)` as a primitive
that does NOT enforce the rebuild combination — the CLI is responsible
for orchestrating "clear → rotate → re-index." Mechanism vs. policy.

### 5. MCP protocol cleanliness

- Use the official `mcp` Python SDK, never hand-rolled.
- Tool input schemas validated via Pydantic. Bad input returns a clean MCP
  error, not a stack trace.
- Long operations (initial index) never run during tool calls. The server
  refuses to serve queries until the index exists; the CLI does the indexing.
- **stdio transport must not write anything to stdout except MCP frames.**
  All logging goes to stderr (Phase 1) or `~/.recall/logs/recall.log`
  (added in Phase 3 by `server.py` as a rotating file handler). Until
  Phase 3 lands, stderr-only is correct. A single stray `print()` silently
  breaks the server. There is a CI test (Phase 3) that spawns the server,
  sends `initialize`, and asserts every line on stdout is a valid MCP
  JSON-RPC frame.

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

## Composition is where bugs live

Two consecutive commits (1.2 and 1.3) found bugs of identical shape:

- **1.2 — `URL_USERINFO` vs `<REDACTED:GITHUB_TOKEN>`.** The userinfo
  regex was correct in isolation. After `GITHUB_TOKEN` had run, it met
  `<REDACTED:GITHUB_TOKEN>` and re-interpreted `<REDACTED` / `:` /
  `GITHUB_TOKEN>` as user / sep / pass — clobbering the GitHub redaction.
  Each pattern was right alone. Fix: exclude `<` and `>` from the userinfo
  char classes.
- **1.3 — `migrate()` wrapping `executescript()` in `transaction()`.**
  In autocommit mode each function was correct alone. Composed,
  `executescript` issued an implicit COMMIT before running, leaving no
  open transaction for `transaction()`'s explicit COMMIT — every test
  using the migrated DB exploded with `cannot commit - no transaction is
  active`. Fix: don't wrap.

**Pattern:** primitives correct in isolation, broken at the boundary.

**Discipline:** when introducing a new primitive that interacts with
existing ones, write the *composition test first*. Don't trust that the
Cartesian product of correct primitives is correct. The shape this rule
catches: regex pack ordering / idempotency, transaction managers around
DDL, signal handlers during async work, decorators stacked unaware of
each other, etc. When you can't easily express the boundary as a test,
the design probably has the wrong split.

When a real composition bug is caught mid-suite, the handoff report MUST
name the root cause + fix + an inline-comment-for-future-contributors
(so the next session sees the trap before re-stepping in it).

## Architectural seams

**source → indexer (streaming OK) → embedder (must batch) → DB write (per-batch).**

Atuin reader streams via cursor, so a million-row atuin DB doesn't blow up
memory. Good. But the embedding step in Phase 2 can't stream — it batches.
Worth knowing the architectural seam: source → indexer (streaming OK) →
embedder (must batch) → DB write (per-batch).

When designing Phase 2 (Commits 2.5–2.8), respect this seam: don't
accidentally collapse it. Specifically:

- Don't `list(source.iter_entries())` upfront — sources are streaming for
  a reason; eagerly materializing kills the memory advantage on large
  atuin DBs.
- Don't embed one entry at a time — sentence-transformers wants batches
  (typically 32–64) for throughput; the per-call overhead dominates
  otherwise. That's the throughput cliff.
- Do batch DB writes per embedding batch — one transaction per N rows
  amortizes commit cost while keeping memory bounded.

## Deferred items (file as GitHub issues at end of Phase 1)

These were called out and intentionally deferred. File them as GitHub
issues when the remote lands at the Phase 1 / Phase 2 boundary, so they
don't get lost in chat history.

- **Scrubber coverage gaps** — tag `v1-launch-blocker`. Highest priority
  adds: Stripe (`sk_live_…`, `pk_live_…`) and npm tokens (`npm_…`).
  Lower priority: Twilio account SIDs, Heroku / Cloudflare API tokens,
  Discord bot tokens, MongoDB connection strings without `://user:pass@`,
  GCP service-account JSON inline, `~/.netrc` / `~/.pgpass` references,
  generic `password:` (colon, no `=`), base64-encoded secrets pasted as
  positional args.
- **Bash multi-line limitation** — low-priority docs. A bash command
  spanning multiple physical lines emits as separate `Entry`s. Document
  in the README (Phase 4). Bash history doesn't preserve enough context
  to disambiguate reliably.
- **Pydantic vs dataclass perf decision** — defer until Phase 2 perf
  tests trigger. `Entry` is a Pydantic v2 model; ~100 µs per build is
  plausible. At 50k entries that's ~5s of pure validation. If the
  `< 60s for 50k index` perf gate fails and pydantic is hot, switch to
  `dataclass(slots=True)` (faster, loses validation) or
  `ConfigDict(validate_default=False)` (cheaper, keeps validation).
- **`RECALL_DB_PATH` user documentation** — Phase 2 todo. The env
  override exists in `db.py` (resolution: arg > env > `~/.recall/
  db.sqlite`) and has tests, but no user-facing surface yet. The
  eventual `recall index` CLI help, README install section, and CLI
  reference need a one-liner each.
- **`[dev]` dep-list sync CI check** — low-priority CI hygiene. Dev
  deps are mirrored across `[project.optional-dependencies].dev` and
  `[dependency-groups].dev` in `pyproject.toml` to support both
  `pip install -e ".[dev]"` and `uv sync`. A 5-line CI script that
  parses `pyproject.toml`, asserts the two dev sets are identical, and
  fails the build on divergence would protect against silent drift in
  PRs that update only one. Comments in `pyproject.toml` warn careful
  readers; this guards against the rest. Address whenever CI is next
  touched.
- **Canonical sentinel CI check** — low-priority CI hygiene, same tier
  as the dep-list sync check. The scrubber fixture
  (`tests/fixtures/secrets_corpus.txt`) uses the canonical sentinel
  `FAKEFAKE` so synthetic tokens cannot match real-world secret patterns
  (per the workflow rule in §"Workflow expectations"). A 5-line CI
  script that scans every non-comment line of the corpus and fails the
  build if any line lacks `FAKEFAKE` would protect against a future PR
  adding a new pattern with realistic-shape synthetic data and re-tripping
  GitHub's secret scanner. Address whenever CI is next touched.
- **Remove embed-lane exit-5 tolerance from `ci.yml`** — Phase 2 task,
  tied to the first `@pytest.mark.embed` test landing. The current
  workflow has a `REMOVE THIS once tests/test_embed.py exists with at
  least one @pytest.mark.embed test` comment around a shell-level
  `[ $rc -eq 5 ]` allowance. Once Phase 2 ships an actual embed test,
  the marker matches something, exit 5 disappears, and the tolerance
  becomes dead weight. Grep for `REMOVE THIS once` to find the line.
- **Bump GitHub Actions to Node 24 versions** — medium-priority CI
  cleanup, deadline 2026-06-02. `actions/checkout`, `astral-sh/setup-uv`,
  and `actions/cache` are running on Node.js 20, which GitHub deprecates
  in favor of Node 24 on June 2, 2026. Two paths: bump action major
  versions when newer ones release with Node 24 support, or opt in
  early via `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` in the workflow
  env block. Don't ignore — full Node 20 removal is 2026-09-16.

The four CI-related deferred items above (`[dev]` dep-list sync check,
canonical sentinel check, embed-lane exit-5 tolerance removal, and the
Node 24 bump) collectively form a **CI cleanup pass** — batch them as
a single set of changes at end of Phase 2 rather than addressing one
at a time. Single PR, single review, single CI run to verify the whole
pass; less context-switching.

## End-of-Phase-1 checklist

Run this checklist at the Phase 1 / Phase 2 boundary, BEFORE starting
Phase 2 work:

- [ ] All four Phase 1 commits landed (1.1, 1.2, 1.3, 1.4); tree is clean.
- [ ] `pytest -k scrub` (canary) and full `pytest` both pass on `main`.
- [ ] Create the GitHub remote and push `main`. The first PR opened
      against this remote will exercise the `scrub-canary` job for real
      (its end-to-end execution has been unverified up to this point —
      see §1). **If the `scrub-canary` job does not appear in the
      Actions tab on that PR, the path filter or trigger is wrong; fix
      before merging anything else.**
- [ ] File deferred items as GitHub issues (see "Deferred items"). Tag
      the scrubber-coverage gap as `v1-launch-blocker`.
- [ ] README skeleton update is a Phase 4 deliverable, but the GitHub
      repo's first impression matters — at minimum, mark "Status:
      pre-alpha, Phase 1 complete" in the readme on first push so
      drive-by visitors aren't confused.

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
- Never commit raw history fixtures with real secrets. Synthesize them
  with the canonical sentinel string `FAKEFAKE` so they cannot match
  real-world secret patterns and cannot trip GitHub's secret scanner.
  A grep for `FAKEFAKE` across `tests/fixtures/secrets_corpus.txt` must
  yield at least one match per non-comment line.
- **GitHub remote / first push is deferred to the end of Phase 1.** The
  first push should land a coherent Phase 1 (foundations + scrubber + db +
  source readers), not a half-built skeleton. Don't create the remote
  earlier even if the prompt to do so is tempting.
