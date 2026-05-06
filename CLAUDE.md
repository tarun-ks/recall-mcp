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

**Real-history validation (2026-05-04, dogfood prep for 2.6.5).**
`scrub.py` was tested against ~1800 lines of real shell history
(1326 zsh + 503 bash) during dogfood candidate selection. **53 real
secrets caught** — primarily URL userinfo (`https://user:pass@…`,
45 instances) and JWTs (12 instances) in saved curl invocations and
auth flows. **Zero credential or PII misses on the commands the
scrubber had touched** (the original 33 redacted commands held).

**The audit also surfaced two coverage gaps in the broader untouched
candidate pool**: (1) personal email addresses in CLI flag values
(e.g. `gcloud config set account <handle>@gmail.com`) — 2 instances;
(2) Python kwarg-form passwords (e.g. `psycopg2.connect(..., password='admin123', ...)`) —
14 instances. Both are real-shape secrets in known-position contexts the
scrubber doesn't currently match. Pattern additions tracked under issue #2
("scrubber-coverage-gaps") as Tier 1 (`v1-launch-blocker`); deliberately
NOT added in 2.6.5 to keep the dogfood commit scoped. The 32-line
synthetic corpus (`tests/fixtures/secrets_corpus.txt`) is the regression-
prevention mechanism; this 1800-line real-history pass is the
validation-on-real-data evidence.

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

**Indexer cursor schema (added 2.8).** The indexer stores per-source
incremental cursors in the `meta` table under keys named
`cursor_<source>` — e.g. `cursor_zsh`, `cursor_bash`, `cursor_atuin`.
Each value is the wall-clock unix-second timestamp of the last entry
successfully committed for that source. **Adding a new source means
adding a new `cursor_<name>` key, not migrating existing data** —
older sources' cursors are unaffected; the new source starts at None
(== "from the beginning").

The cursor advances atomically with the row inserts it represents
(within the same transaction). On crash mid-batch, the cursor stays
at the last successfully committed batch's max ts — the next
indexing run re-processes only entries the previous run hadn't
committed. Entries with `ts = 0` (unknown timestamp) are always
yielded by sources regardless of cursor and never advance the cursor;
the `UNIQUE(source, text_hash, ts)` constraint backstops dedup.

### 3. atuin schema robustness

Open the user's atuin database with `?mode=ro&immutable=1` so we never write
to it. Detect schema by checking for required columns at runtime, not by
hardcoded column order. Always `SELECT` by name.

Validated empirically: the project's own author's machine has zsh + bash
but no atuin during 2.6.5 dogfood selection.

### 4. Embedding consistency

Store embedding model name + revision in the `meta` table on first index.
On startup, refuse queries if the configured model differs from the indexed
one — prompt the user to run `recall index --rebuild`. Mixing embeddings
across models silently produces garbage results.

### 4a. Embedder public API contract

`src/recall/embed.py`'s public surface is **frozen at Commit 2.5**. The 2.7
"production-grade" rewrite changes internals only — MPS warmup, explicit
batch-size, opt-in query cache (in SemanticRanker, not Embedder) — under the
same public interface. Eval numbers must match within **±0.01 recall@5 noise
band**, which is the contract that makes "behavior-preserving rewrite"
testable rather than vibes.

Frozen public surface (with the 2.7 kwarg addition):

```python
class Embedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        model_revision: str | None = None,
        cache_folder: Path | None = None,
        batch_size: int | None = None,    # added 2.7
    ) -> None: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        # shape (len(texts), dim); float32; L2-normalized
        ...

    dim: int                         # read-only after __init__
    model_name: str                  # the id passed at construction
    model_revision: str | None       # pinned revision, or None
```

Anything else (private methods `_*`, internal caching, batching strategy,
device selection) is implementation detail and may change. Tests that lock
down behavior should test the public surface only.

**Frozen API extension policy.** The Embedder public API
(`__init__`, `encode`, `dim`, `model_name`, `model_revision`) is closed
to breaking changes — no removals, no signature changes to existing
parameters, no behavioral changes to `encode(texts) -> np.ndarray`.
Optional kwargs MAY be added to `__init__` and `encode` if (a) they
preserve all existing call sites, (b) defaults reproduce
pre-extension behavior, (c) the rationale is documented in the commit
landing the addition. The 2.7 commit added `batch_size: int | None =
None` under this rule. Future additions follow the same gate.

**Performance contract (2.7 → 2.7.5).** Steady-state warm runtime
targets on the eval lane (semantic ranker on nl2bash, corpus 10,624
/ queries 11,348):

- M-series Mac (warm): **≤17s** at 2.7.5 (was ≤26s at 2.7; achieved
  16.25s median post-2.7.5 via the sqlite-vec → numpy matmul
  rewrite). Search stage dropped from 13.97s → 5.86s.
- M-series Mac (first-shell-invocation, cold MPS): ≤17s — the MPS
  warmup at `Embedder.__init__` amortizes the kernel-compilation cost
- GitHub-hosted Linux (CPU, no MPS): **≤220s** at 2.7.5 (was ≤260s
  at 2.7 deferred; original ≤195s aspiration revised — see
  "Architectural floor finding (2.7.5)" below). Achieved 219.47s on
  CI verify branch: search stage 113.38s, down from 149.76s at 2.7
  (−36.4s — the matmul win).

**Architectural floor finding (2.7.5).** The original ≤195s Linux
target assumed sqlite-vec MATCH overhead was ~80s of the 156s search
stage. Empirically (CI verify branch) it was ~36s — matmul replacement
saved 36s exactly, but the remaining 113s is encoder-bound (query
encoding ~110s + matmul ~3s). Linux floor for 2.7.5's scope is
therefore **~218s**: 5s init + 100s corpus encoding + 110s query
encoding + 3s matmul. We landed at 219.47s, ~1.5s above the floor.
**Below the floor is encoder territory** — different model,
parallel/threaded encoding, or different runtime. Not 2.7.5's scope;
not on 2.8's critical path either. The gate is updated to ≤220s to
reflect the empirically-determined floor; further Linux throughput
work would be its own commit if/when the cost becomes load-bearing.

This is the second project-level "constraint surfacing" finding in
two consecutive commits (2.7's platform-divergence finding was the
first). Pattern: budget-tightening exposes the next dominant cost;
naming it explicitly lets subsequent planning rounds inherit the
right picture rather than re-discovering it.

**M-series gate recalibration (≤24s → ≤26s).** Reflects the platform-
divergence finding documented in §6 "Platform-divergent optimal batch
sizes": batch=128 minimizes M-series time at 24.4s but blows up Linux
to 289s; batch=32 minimizes Linux but blows up M-series to 44.6s. The
platform-balanced default of batch=64 lands M-series at ~23s and Linux
approximately at baseline ~250s. The remaining gap to the originally-
targeted Linux ≤195s is exclusively 2.7.5's job, not 2.7's.

**Linux ≤195s gate explicitly deferred.** 2.7's encode-side
improvements alone cannot reach Linux ≤195s; the search-stage
sqlite-vec replacement in 2.7.5 is the load-bearing change. 2.7 ships
approximately Linux-neutral with the platform-balanced batch=64
default. The Linux throughput gate remains live but is addressed by
2.7.5, not by 2.7.

**Earlier ≤17s framing was over-calibrated** against a stage-blind
11.2× platform translation factor. Per-stage decomposition shows
encoding is ~17× slower on Linux while sqlite-vec MATCH is ~12×
slower. On M-series specifically, encoding is already cheap and
per-query sqlite-vec is the dominant cost (13.5s of 24.4s total,
55%). 2.7's encode-side improvements therefore cap at ~1s saved on
M-series — the gate is decoupled from Linux's by per-stage cost
composition, not by a uniform translation factor.

The batch size is a single tunable: `RECALL_EMBED_BATCH_SIZE` env var
or `Embedder(batch_size=…)` kwarg, default 64. Per-platform tuning
available for advanced users; default 64 is the cross-platform
compromise validated at 2.7.

**Behavior-preservation gate.** `tests/test_embed_behavior_preservation.py`
pins nl2bash semantic recall@5 to the bit-identical 2.5/2.6 baseline
`0.44862530842439197` within `TOLERANCE=0.0001`. The test logs the
delta on every run so a non-zero magnitude (float-noise from batching
changes, etc.) is visible to reviewers even when the gate passes.

**Equivalence-test contract (added 2.7.5; revised 2.7.5-hotfix).**
Top-5 IDs from `SemanticRanker` are pinned to the **deterministic
numpy algorithm's** canonical output across all 11,348 nl2bash queries.
The fixture is now the deterministic algorithm's reference, not
sqlite-vec's. Cross-runner determinism is the contract; any divergence
across runners is a real algorithmic regression, not float32 noise.

**Tie-breaking convention (2.7.5-hotfix).** Lower-index wins on ties.
Implemented via composite-key argsort over `-rounded_score + index *
1e-10`, where score is rounded to 5-decimal precision (1e-5) to absorb
BLAS epsilon variance. Rationale: matches natural sort intuition;
matches empirical Linux CI convergence under the prior argpartition
implementation; eliminates argpartition's "unspecified order among
equal elements" non-determinism that broke cross-runner tests.

The decision matrix in `tests/test_retrieve_semantic_equivalence.py`
is the gate:

| outcome | verdict (post-hotfix) |
|---|---|
| set equality holds + recall@5 ±0.0001 | algorithms equivalent (normal; the expected case) |
| set miss + recall@5 holds | pre-hotfix this was "tie reordering acceptable"; post-hotfix should NOT occur — the algorithm is deterministic + the fixture is pinned to it. Surface for investigation. |
| recall@5 drift + set holds | impossible by construction (set equality implies same gold hits) |
| both fail | real algorithmic bug |

The fixture lives at `tests/fixtures/nl2bash_sqlite_vec_top5.json`
(filename retained from 2.7.5 era for git-history continuity; the
`_meta.provenance` field documents what it actually is now).
Regenerating it requires explicit reasoning about why the reference
shifts (e.g. embedder model change, corpus change, tie-break convention
change). `test_fixture_provenance_is_pinned` enforces the metadata
schema.

**Baseline shifts ARE landmark events.** The pinned recall@5 baseline
in `test_embed_behavior_preservation.py` shifted at 2.7.5-hotfix from
0.44862530842439197 (sqlite-vec / accidental-argpartition) to
0.44836094465985193 (deterministic numpy). The shift is **de-aliasing,
not regression**: the old value was an artifact of platform-specific
tie-breaking that happened to match sqlite-vec; the new value is what
the deterministic algorithm produces canonically. Future commits are
gated against the new anchor with the same ±0.0001 tolerance.

### 4b. Dedup salt and rebuild policy

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
- Initial index of 50k commands: < 60 s **on M-series Mac only**
  (added 2.8). Indexer is encode-bound on Linux per the 2.7
  platform-divergence finding; Linux indexing throughput is bounded
  by Embedder throughput, which is itself constrained by the
  platform-balanced batch=64 default (CLAUDE.md §6 "Platform-divergent
  optimal batch sizes"). Linux empirical number recorded post-impl;
  no Linux gate enforced — the constraint chain runs through the
  Embedder, not the indexer.
- Eval harness (`recall eval --dataset nl2bash`): ≤ 60 s on M-series Mac
  (target). Soft warning printed at 90 s; hard fail (raised by the harness
  itself, not just CI) at 120 s. The hard fail makes the runtime gate live
  in the harness, so an accidental 10× slowdown trips the discipline before
  CI even sees it.

**Platform-translation factor (CI vs M-series).** CI Linux is
**~11× slower than M-series Mac** for encode-bound workloads, not the
previously documented 7×. Recalibrated in 2.7 against actual measurements
(Linux 259.66s / M-series 23.18s = 11.2× on nl2bash semantic). The factor
varies by stage: model load ~1×, encoding ~14×, search loop ~12×.

**Cold vs warm.** `eval/results.json` records steady-state warm runs.
First-shell-invocation costs ~14s additional MPS kernel-compilation on
M-series; the MPS warmup at `Embedder.__init__` (added 2.7) amortizes
this so user-facing first-call latency reflects the steady-state
number. CI Linux has no MPS warmup cost; the warmup call is a fast
no-op there.

**CI runner-to-runner variance (added 2.8).** Semantic CI Linux runtime
exhibits **~15-20% runner-to-runner variance** (observed range
230-276s across three runs as of 2.8). Budgets are sized against the
upper end; single-run shifts within this range are noise, not
regression. **Future commits with retrieval changes should expect
40-50s variance per run** and not over-react to one number. If you need
a real signal on whether a change moved the bar, run CI 3+ times and
look at the median, not a single point.

**Platform-divergent optimal batch sizes (2.7 finding).** Optimal
batch size diverges by platform: M-series MPS prefers batch=128
(kernel-launch amortization wins; 333 launches at batch=32 vs 84 at
batch=128 maps to a 3.5× index slowdown). Linux CPU prefers batch=32
(sentence-transformers' tuned default; cache-pressure avoidance). The
static default of 64 is the platform-balanced compromise, accepting
~1s of M-series cost (24.4s → ~25.5s) and ~5% Linux improvement vs
effective pre-2.7 batch=32 (in trade for stable behavior at batch=128
on M-series turning into a 30s regression on Linux). Per-platform
tuning via `RECALL_EMBED_BATCH_SIZE` env var available for advanced
users.

The finding itself — that a single static default cannot satisfy both
platforms — is the load-bearing insight, not the specific 64. Future
phases that pick batch sizes (the indexer in 2.8, the MCP server in
Phase 3) should make the same per-platform tuning available rather
than baking a single number into call sites.

**Thermal sensitivity (M-series).** Sustained heavy MPS work (e.g.
back-to-back eval runs at varying batch sizes) can trigger firmware
down-clocking on M-series Macs, contaminating throughput
measurements. Verified empirically at 2.7: batch=64 median across 3
runs was 25.5s under cool conditions but 31.3s on a session that had
just run 6 prior measurements. Cooldown protocol when collecting
clean numbers: 15-20 minute idle, then a quick warmup probe (encode
~100 strings at the target batch size; <1s post-init confirms cool).
If the warmup probe runs slow, system is still hot — wait longer.

Tests live at `tests/test_perf.py`.

### 7. Eval harness must run

`recall eval --dataset nl2bash` builds a fresh in-memory index from nl2bash
commands, runs all NL queries against five rankers (semantic + four lexical
baselines), reports recall@1 / recall@5 / MRR per ranker.

**Calibrated baseline (Commit 2.6, bge-small-en-v1.5):**

| ranker | recall@1 | recall@5 | MRR | semantic Δ |
| --- | --- | --- | --- | --- |
| semantic | 0.2893 | 0.4486 | 0.3507 | (baseline) |
| bm25 | 0.2558 | 0.4047 | 0.3122 | 1.11× |
| token-overlap | 0.1610 | 0.3072 | 0.2146 | 1.46× |
| fuzzy | 0.0226 | 0.0963 | 0.0490 | 4.66× |
| naive | 0.0396 | 0.0857 | 0.0565 | 5.23× |

Random-baseline recall@5: 0.0005 (≈ 5/10624).

The recall@5 = 0.4486 number is empirical, not aspirational. The original
brief wrote "recall@5 > 0.75" — that was aspirational language;
bge-small-en-v1.5 at default settings lands at 0.45 on `nl2bash`.

**Paraphrastic-bias note (design feature, not caveat).** nl2bash is
constructed to test paraphrase handling — many NL queries are
paraphrases of the same intent, mapping to commands with different
surface forms. Lexical baselines being weak on paraphrastic queries is
the dataset measuring exactly what semantic retrieval is for. The
"naive" / "fuzzy" rankers landing at recall@5 ~ 0.09 isn't a flaw of
the eval; it's the eval doing its job. Conversely, BM25 being
relatively strong (0.4047) on nl2bash reflects BM25's IDF weighting
handling rare-term paraphrases well — an honest IR baseline, not a
strawman.

**Headline pitch is delta-vs-lexical-baseline, not absolute recall@5.**
Decision tree for v1 model choice (three-way, after Commit 2.6 numbers
landed at 1.11× vs the strongest lexical):

| semantic vs strongest lexical baseline on nl2bash | action |
| --- | --- |
| ≥ 2× | ship bge-small-en-v1.5 in v1; the delta is the value-prop |
| 1.5× – 2× | evaluate bge-base before v1 (mid-size step) |
| < 1.5× | evaluate bge-large or gte-large directly; bge-base is unlikely to be enough given the gap |

**Commit 2.6's number lands in the bottom bucket (1.11× vs BM25).** The
next model experiment should be bge-large or gte-large (skipping
bge-base, which is unlikely to close the gap). This decision is
*conditional on data we don't have yet* — see "next experiments" below.

**Two pending experiments before any v1 model decision:**

1. **Dogfood numbers on real shell history.** Lexical noise in actual
   shell history (typos, project-specific jargon, abbreviations) is
   plausibly higher than in `nl2bash`'s curated NL paraphrases. The
   semantic vs lexical delta may widen meaningfully on dogfood and
   narrow in nl2bash's favor. Both numbers are required for the v1
   model decision; nl2bash alone is insufficient. Lands in a follow-up
   commit (call it 2.6.5) immediately after 2.6.
2. **bge-large or gte-large head-to-head.** Single-run comparison of
   the larger embedders against bge-small on the same eval harness.
   If a larger model materially closes the 1.11× gap on `nl2bash` AND
   improves dogfood numbers, ship the larger one. If not, the v1 story
   shifts toward "fast local CPU semantic with modest delta over BM25"
   rather than "semantic crushes lexical."

These numbers ship in the README at v1, alongside the lexical-delta
table — the absolute recall@5 alone is uninterpretable to a drive-by
visitor; the delta tells the value story.

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

**Phase 2 — core retrieval** *(reordered from original brief: eval moves to 2.5
because the project's value proposition is a probabilistic claim, and three
commits of optimization on an unverified premise is three commits in an
arbitrary direction — see "Phase 2 gating rules" below)*
- 2.5 **Eval harness + minimal `embed.py`** (sets the measuring stick;
      produces the first real recall@k numbers on `nl2bash`)
- 2.6 Substring-grep baseline in the same harness; the delta vs 2.5 is the
      value-proposition expressed as a number
- 2.7 `embed.py` production-grade — MPS warmup, explicit batch-size,
      opt-in query cache, frozen API extension policy. **Behavior-
      preserving rewrite** (eval numbers must match within ±0.01
      recall@5 noise band; 2.7 achieved bit-identical delta = 0.0).
      Linux throughput gate explicitly deferred to 2.7.5 per the
      platform-divergence finding (CLAUDE.md §6).
- 2.7.5 **Semantic search loop: per-query sqlite-vec → batched numpy
      matmul.** The Linux ≤195s gate's load-bearing change. 2.7's
      measurement validated this scope: per-query sqlite-vec MATCH
      overhead has been the dominant search-stage cost since 2.5
      (~80s of CI Linux's 156s search stage). Empirical-equivalence
      test required: batched matmul argpartition vs per-query
      sqlite-vec MATCH must produce identical top-5 IDs across
      nl2bash. Same ±0.0001 recall@5 behavior gate as 2.7. **Binding
      before 2.8 starts** — indexer adds eval-lane content that
      compounds existing CI pressure.
- 2.8 Indexer: `HistorySource` → scrubber → embedder → DB write,
      respecting the architectural seam. Per-source ts cursor in
      `meta`; 1024-row indexer batches feeding the embedder's
      internal batch_size=64; `--rebuild` does DROP + CREATE
      (well-tested vec0 reset path); `--new-salt` requires `--rebuild`.
      Adversarial scrubber-integration test asserts zero secret-pattern
      matches in `commands.text_scrubbed` after indexing the
      synthetic-secrets corpus. **Indexer is consumer-less until Phase
      3's MCP server lands** — integration tests verify DB-layer
      correctness; user-facing round-trip ("index → search via MCP
      tool") is gated on Phase 3 and added then.
- 2.9 Hybrid search (vector + FTS5 with RRF k=60) — **DEFERRED TO
      v1.1.** Original Phase 2 plan put this before Phase 3; at 2.8
      closure the decision was to defer based on (a) 4× dogfood and
      1.11× nl2bash semantic-only deltas suggesting hybrid is
      marginal-gain, (b) Phase 3 produces actual user-feedback data
      that should inform whether hybrid is worth shipping, (c) v1's
      launch story doesn't depend on it. See deferred-items entry
      "2.9 hybrid search" for the v1.1 implementation framing.

**Phase 3 — MCP surface (5-commit split, locked Phase-3 §10)**
- 3.9  Server skeleton: lifecycle, DB read-only connection, embedder
       lazy-load, stale-index check, empty tool registry. The MCP
       protocol bootstraps (initialize, list_tools=[]); no tools yet.
- 3.10 Six tool implementations + Pydantic input/output schemas
       (CommandHit, SequenceHit, CommandStats per locked Q2). Adds
       runtime stdout-redirect defense around embedder.encode.
- 3.11 stdio cleanliness test suite: subprocess test + ruff T201 +
       custom AST check. Cold-cache stress case included. Runs on
       every PR + main push + scheduled daily CI.
- 3.12 Pseudo-client + recorded-session-fixture replay tests.
       Includes "long idle, then tool call" recorded scenario.
- 3.13 Manual smoke-test checklist + docs/clients-tested.md
       (Claude Desktop macOS+Windows, Cursor macOS, Zed macOS,
       Cline VS Code macOS). The v1 launch-readiness gate.

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

## Phase 2 gating rules

Phase 2 ships retrieval — a probabilistic system. Phase 1's binary tests-
pass discipline is necessary but not sufficient: code can be correct in
every unit test and still ship a regression in retrieval quality. Add a
probabilistic gate alongside.

**Every commit that touches retrieval logic must report:**
1. Current `recall@5` on each eval dataset (`nl2bash`, eventually `dogfood`).
2. Delta vs the prior commit's number (`recall@5 = 0.78 (was 0.76, +0.02)`).
3. The number lives in three places:
   - **Commit message** — headline, single line, at-a-glance history
   - **`eval/results.json`** — append-only history of every run; one record
     per `recall eval` invocation, keyed by `commit_sha + dataset`. Schema
     per record: `{commit_sha, timestamp, dataset, recall_at_1, recall_at_5,
     mrr, runtime_seconds, runtime_breakdown, model_name, model_revision,
     random_baseline_recall_at_5, n_queries, n_corpus}`. Longitudinal view
     beats single snapshot for spotting drift later.
   - **CI logs** — the embed lane runs `recall eval` on every PR; the
     regression gate (`.github/scripts/check_eval_regression.py`) compares
     PR's just-computed number against `origin/main:eval/results.json`'s
     most recent record for the same dataset. **PR fails if recall@5
     dropped by more than the noise band (±0.01).**

**Noise band.** `±0.01 recall@5`. This is what makes both the
"behavior-preserving rewrite" of `embed.py` (2.5→2.7) testable AND the
regression gate enforceable. Tighter would be flaky; looser would let real
regressions through.

**Probabilistic ≠ unrigorous.** Build the measuring stick before the things
being measured. Iterating optimization commits without an eval harness is
iterating blind; you discover at the end whether the project has any
reason to exist. Eval first, then optimize against it.

**Dogfood vs benchmark.** `nl2bash` provides public-benchmark credibility
(numbers comparable to published results). The dogfood set (real queries
from tarun's actual zsh history, lands in a follow-up commit) provides
real-use-quality evidence. Both numbers must move together for the project
to be worth shipping; one without the other is a partial picture.

## Composition is where bugs live

Recurring pattern across this project — primitives correct in isolation,
broken at the boundary. Named instances:

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
- **2.5 — Typer single-command auto-collapse.** A `typer.Typer` app with
  exactly one `@app.command(...)` registered treats that command as the
  default — `recall eval --dataset nl2bash` then fails with
  `unexpected extra argument (eval)` because typer never expected a
  subcommand name. The bug was caught by *running the actual CLI*, not
  by import-time tests. **Lesson: imports passing ≠ behavior correct.
  Framework behavior can change based on registration count, plugin
  presence, or env state. Integration runs (actually invoking the
  CLI/server) catch what import-time tests miss.** Fix: register stub
  commands for `index` (2.8) and `serve` (Phase 3) so typer always sees
  ≥ 2 commands and never collapses.
- **2.5 — Lazy-import test-collection cost.** A top-level
  `from sentence_transformers import SentenceTransformer` in
  `src/recall/embed.py` made `pytest -k scrub` (the canary) go from
  ~50 ms to ~30 s — pytest's collection imports every `tests/test_*.py`
  file, including `test_eval.py`, which transitively pulled torch
  (~25 s of import overhead). No test ever ran torch; the import alone
  destroyed scrub-canary's value. **Lesson: test collection time is a
  project asset. The scrub canary's value depends on sub-second runtime;
  a top-level torch import in any `tests/test_*.py` file silently
  destroys that even if no test ever runs torch. New deps imported
  during pytest collection must not exceed a budget — suggest scrub
  canary + smoke ≤ 1s total. Lazy-import any heavy ML deps via
  `if TYPE_CHECKING:` for type hints + in-method imports for runtime.**
  Fix: move `SentenceTransformer` import inside `Embedder.__init__`,
  use `if TYPE_CHECKING:` for annotations.
- **2.6 — C-backed library performance: kernel vs dispatch.**
  ``rapidfuzz.process.extract`` per-query was 329 s for fuzzy retrieval
  over nl2bash. ``rapidfuzz.process.cdist(..., workers=-1)`` brought
  it to 31 s — a 9× speedup that comes entirely from running the C
  kernel multi-threaded on M-series cores instead of single-threaded.
  The first pattern dispatches into the C kernel ~11k times (one per
  query) with Python-loop overhead between dispatches; the second
  hands the whole 11k × 10k pairwise matrix to one C call. **Lesson:
  when integrating a C-backed library, verify the bottleneck is in
  the C kernel, not in the Python dispatch layer above it.
  Performance docs typically describe kernel throughput; the actual
  code path may be Python-bound and the benchmark numbers won't apply
  unless you batch.** Fix: use the library's batch / matrix /
  parallel API, not the per-item one. (See ``recall.retrieve.fuzzy``.)
- **2.6 — `multiprocessing.Pool` for Python-loop-bound rankers.**
  Naive substring at 85 s single-threaded → 11 s with a process pool
  (helper function module-level for picklability — same pattern
  that's now in ``recall.retrieve.substring._score_chunk``). For pure
  Python ranking loops there's no C kernel to lean on; the only
  speedup is multi-process parallelism. Token-overlap and similar
  pure-Python rankers should reach for the same pattern when their
  scale grows.
- **2.7 — hardcoded default value baked execution-context
  assumption into source.** `HARD_RUNTIME_FAIL_S = 120` in
  `recall.eval.runner` was a sensible default for local M-series
  development at 2.5 (where eval ran in ~25s). When 2.7 added a
  `@pytest.mark.embed`-marked behavior-preservation test that
  invoked `run_eval()` from inside `pytest -m embed`, the test
  inherited the wrong execution context: CI Linux semantic eval
  runs ~260s, well over the 120s default, but the workflow's env
  override (`RECALL_EVAL_HARD_FAIL_S: "360"`) was scoped only to
  the `recall eval` CLI step, not the pytest step. The new caller
  silently inherited the wrong context. **Lesson: a hardcoded
  default value bakes an execution-context assumption into source
  code; when a new caller from a different context appears, the
  default is wrong but invisibly so. The one-line workflow fix
  (add the env to the pytest step) patches the symptom; the
  deeper fix is making defaults context-aware** (filed as
  deferred-items entry "HARD_RUNTIME_FAIL_S default should be
  context-aware"). Surfaced when 2.7's verify-branch CI failed
  the behavior-preservation test on runtime, not on recall@5.
- **2.7.5 — argpartition tie-handling under BLAS-induced epsilon
  variance produces non-deterministic top-k across CI runners.**
  Primitives correct in isolation (BLAS matmul, np.argpartition);
  composing into observable instability at the test gate.

  BLAS matmul rounding order produces tiny per-element variance
  (~1e-7 on individual cosine scores) — different CI runners,
  different thread interleavings, identical inputs. argpartition's
  "unspecified order among equal elements" semantic amplifies that
  1e-7 input variance into top-5 set divergence at near-tied score
  boundaries — and that propagates to recall@5 drift past the
  ±0.0001 tolerance. Verify-branch CI passed at delta -1e-04;
  post-merge main CI on identical content failed at -2.6e-04.

  **Fix: composite-key argsort eliminates argpartition's tie
  ambiguity** (`-rounded_score + index * 1e-10`, with rounding to
  1e-5 to absorb BLAS variance below meaningful score precision).
  Performance cost ~+0.5s on M-series (full O(n log n) argsort
  instead of O(n) argpartition), well within the ≤17s gate.

  **De-aliasing sub-note (2.7.5-hotfix iteration finding).** The
  initial fix attempt (lexsort over composite key) achieved
  determinism but produced different recall@5 than the original
  sqlite-vec baseline because sqlite-vec's tie-breaking convention
  is implementation-specific and not index-order. The original
  baseline (0.44862530842439197) wasn't a "true" recall@5; it was
  argpartition's accidental tie-break choice on M-series happening
  to coincide with sqlite-vec's internal ordering on ~39 tie-
  affected queries. We re-anchored the baseline to the deterministic
  algorithm's canonical output (0.44836094465985193) rather than
  attempting to reproduce sqlite-vec's tie-breaking. The shift is
  de-aliasing, not regression.

  Lesson: **equivalence-test fixtures should be pinned to your own
  algorithm's deterministic output, not to a reference implementation
  with different tie-breaking semantics.** Pinning to a foreign
  reference produces ongoing fixture noise (the algorithm and the
  reference will drift in tie-breaks for any ranking near the top-k
  boundary) rather than future-drift signal (where you actually want
  to detect changes in your own algorithm's behavior).

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

## Audit layers compound

Each independent audit pass over the scrubber surfaces gaps the previous
passes didn't. The synthetic 119-test corpus catches regex composition
bugs (URL_USERINFO vs `<REDACTED:...>`); the real-history audit
validates existing patterns hold (53 catches, zero misses on touched
commands); the dogfood selection over an untouched pool surfaced two
gap classes neither prior pass anticipated (personal-email-in-flag-
value, Python kwarg-form passwords). Implication: a single audit layer
is insufficient evidence of completeness; future scrubber additions
should be validated against fresh untouched corpus pulls, not just the
existing test fixture.

## Eval harness

`src/recall/eval/` houses the harness. CLI:

```bash
recall eval --dataset nl2bash                       # all five rankers (default)
recall eval --dataset nl2bash --ranker semantic     # one ranker for iteration
recall eval --dataset nl2bash --ranker bm25
recall eval --dataset nl2bash --no-append           # prints only; no history write
recall eval --dataset nl2bash --output PATH         # custom history file (CI uses /tmp)
```

**Five rankers** (all five run by default per Phase 2 cadence — every
retrieval-touching commit reports all five recall@5 numbers):

| ranker | what it is | source |
| --- | --- | --- |
| `semantic` | bge-small-en-v1.5 + sqlite-vec KNN — the value prop | `recall.retrieve.semantic` |
| `naive` | whitespace-split query words as substrings, count-ranked | `recall.retrieve.substring` |
| `token-overlap` | FTS5-tokenized intersection-count ranking | `recall.retrieve.token_overlap` |
| `bm25` | FTS5's `bm25()` over `unicode61 remove_diacritics 1` tokens | `recall.retrieve.bm25` |
| `fuzzy` | rapidfuzz `partial_ratio` (fzf-like, not exactly fzf) | `recall.retrieve.fuzzy` |

**FTS5 tokenizer is the shared source of truth** for `token-overlap`
and `bm25`. BM25 uses SQLite FTS5's built-in `unicode61
remove_diacritics 1` tokenizer directly. Token-overlap uses
`recall.retrieve.base.fts5_unicode61_tokenize`, a Python-side
approximation of the same tokenizer (NFKD normalization + drop
combining diacritics + casefold + alphanumeric token regex). The two
agree on virtually all practical inputs; rare Unicode edge cases
(scripts whose Letter/Digit classification differs slightly between
Python's `re` and SQLite's table) may produce a token here or there
that's in one but not the other. The agreement is what makes the two
numbers directly comparable on the same corpus.

**`fuzzy` is fzf-*like*, not exactly fzf.** Real fzf adds bonuses for
word-boundary and camelCase matches that `rapidfuzz.partial_ratio`
doesn't model. The user-facing claim is "we beat what zsh+fzf users
do today," not "we beat fzf's exact scoring algorithm." If we ever
need higher fidelity, we can shell out to the fzf binary or evaluate
`pyfzf` — that's a Phase 4 polish concern, not a 2.6 blocker.

**Datasets:** `nl2bash` is shipped at 2.5. The dogfood dataset (real
zsh-history queries from the maintainer) lands in 2.6.5 immediately
after 2.6 — the 1.11× semantic-vs-BM25 finding makes dogfood numbers
materially more important than they were before that delta landed.

**What it does.** Builds an in-memory `:memory:` index per ranker
(SemanticRanker uses sqlite-vec; Bm25Ranker uses FTS5; the others use
plain Python state). For each ranker: instantiate, index the corpus,
search every query, compute recall@1 / recall@5 / MRR with **multi-
reference semantics** ("any gold-reference command in top k counts as
a hit"). Reports a runtime breakdown per ranker (`init`, `index`,
`search`, `total`) — so first-run-with-model-download vs cached
semantic init is legible, not buried in a single 60s number. Also
prints the random baseline (`avg_gold * k_max / n_corpus`) so the
absolute recall@5 number's magnitude is interpretable.

**Runtime gates — two separate budgets:**

*Per-ranker hard fail* — anomaly detector for any single ranker:
- soft target ≤ 60 s on M-series for one ranker
- hard fail at 120 s (`EvalRuntimeError` from runner.py)
- `RECALL_EVAL_HARD_FAIL_S` env override (CI sets 360s)

*All-rankers cumulative* — aggregate across `--ranker all`:
- soft warning at 120 s on M-series (printed by CLI)
- hard fail at 180 s (CLI raises)
- `RECALL_EVAL_ALL_HARD_FAIL_S` env override (CI sets 600s)

The two-budget split exists because per-ranker tightness (an
anomaly detector for "did one ranker suddenly 10×?") is different
from cumulative tightness (the aggregate budget for the all-rankers
loop). Empirical M-series wall-clock for `--ranker all` is ~120-130s
warm; the 180s hard fail gives ~50% headroom against drift. CI
Linux is ~3× slower for the lexical rankers and ~7× for embedding;
600s ceiling preserves the same 50% headroom rule there.

The hard fails in both runner.py (per-ranker) and CLI (cumulative)
mean an accidental 10× slowdown trips the discipline before CI even
sees it. The env-var overrides exist for CI's runner-class delta
specifically — NOT for raising the local discipline ceiling. If you
find yourself wanting to raise the local ceiling, that's a signal
something else is wrong (a regression, a bigger model, a
parallelization regression, etc.) — investigate before adjusting.

**Test coverage exemption: the cumulative-budget logic in
`recall.cli` (the `RECALL_EVAL_ALL_HARD_FAIL_S` handling and the
`time.perf_counter()` cumulative check across the all-rankers loop)
is intentionally NOT covered by automated tests.** Exercising it
end-to-end requires a slow eval invocation; mocking the timing or
fabricating ranker delays would test the mock, not the production
code path. Manual verification only — confirmed at Commit 2.6 by
running `recall eval --ranker all` and observing the soft-warn fire
at ~130 s and the hard-fail correctly raised when budget was lower.
Named here explicitly so a future auditor doesn't wonder why this
branch isn't covered.

**Drafting paraphrastic NL queries.** Drafting NL queries that
genuinely test paraphrase requires explicit discipline. Default
human description of a command tends to re-use the command's content
vocabulary, which lets lexical baselines win on what should be
semantic tests. The 2.6.5 dogfood set surfaced this: 3 of 5
first-draft NLs were vocabulary-overlapping with their commands at
the L tier when intended as M / H. This is a class of subtle eval
bias that would otherwise contaminate any future dogfood expansion
or peer-contributed query sets. **Future dogfood expansion or
peer-contributed query sets must validate paraphrase distance via
shared-token analysis (after FTS5 tokenization) before being tagged
at M or H.** Manual today; the deferred-items entry "NL paraphrase-
distance validator" is the automated guardrail that closes this gap
at scale.

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

**Eval vs production-indexer KNN divergence (added 2.7.5).** The
search-time data structure differs by use case, even though the
embedder and the corpus contents are shared:

- **Eval path** (`recall.retrieve.semantic.SemanticRanker`):
  source → embedder → in-memory numpy ndarray → matmul + argpartition.
  No sqlite-vec, no DB. The corpus is held as `np.ndarray` and search
  is one BLAS call. Eval workloads are small N (≤ ~50k), in-memory,
  repeated across runs — pure-numpy simplicity wins.
- **Production indexer path** (Commit 2.8, future): persistent on-disk
  sqlite-vec virtual table for KNN over 50k+ commands across recall
  invocations. Production workloads are large N, persistent storage,
  single-query latency — sqlite-vec's on-disk index earns its keep
  here. The 482 MB peak memory of full matmul on nl2bash would be
  ~46 GB on a million-row corpus; chunking helps but persistent
  KNN is the right primitive.

Both paths share `Embedder`. They diverge below it. 2.8's planning
round inherits this divergence as a given: don't try to unify the
two; pick the right structure per layer.

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
- **Eval all-rankers wall-clock tension with 2.7's batching** —
  Phase 2 perf-budget concern. Empirical M-series warm runtime for
  `recall eval --ranker all` is ~115 s; the all-rankers cumulative
  hard fail is 180 s — only 65 s of headroom. 2.7's batching
  optimization on `embed.py` is the natural place to win headroom
  back (semantic init/index/search currently dominates ~40 s of the
  total). Tag: tech-debt, performance, phase-2.

  **Throughput-gate fallback chain** (if any all-rankers cumulative
  gate fails after iteration): (1) drop `naive` from `--ranker all`
  default — recall@5 = 0.0857 on nl2bash is the trivial-floor
  baseline, contributing minimal evaluative signal; (2) bump
  `RECALL_EVAL_ALL_HARD_FAIL_S` to 720s with documented justification;
  (3) drop `fuzzy` as last resort — the 4× dogfood framing relies on
  fuzzy as the zsh+fzf comparison; dropping weakens v1 narrative.

  **Status (2.7.5-hotfix):** option (1) executed. The hotfix's
  deterministic-tie-breaking structural fix (full argsort vs
  argpartition) adds ~57s to semantic's CI Linux runtime; pushed
  cumulative wall-clock past the 600s ceiling. Dropped naive from
  `_DEFAULT_RANKER_ORDER` in `cli.py`. `recall eval --ranker naive`
  still works explicitly — it's just not on the default critical-
  path eval. naive's calibrated 0.0857 baseline stays locked in the
  ranker table; if it ever needs re-measurement (e.g. v1 README
  rewrite, or a change suspected of affecting it), use explicit
  `recall eval --ranker naive`. The path stays available; it's just
  not on the default critical-path eval.

  Options (2) and (3) remain unused. **Note:** the 2.7.5-hotfix
  also bumped `RECALL_EVAL_ALL_HARD_FAIL_S` from 600s to **660s**
  (not the chain option-2 emergency relief value of 720s). This is
  separate-concern recalibration, not chain execution: the
  structural-fix cost is permanent and non-reversible without
  sacrificing determinism, so the ceiling reflects the new steady-
  state honestly. Net post-hotfix Linux: ~537s baseline + ~120s
  headroom against drift / future eval-lane additions.

  (Commit 2.7.5 — semantic search loop rewrite — was previously
  listed here as a deferred-items entry. Landed at 2.7.5 as a
  first-class build-order step; see "Phase 2 — core retrieval" → 2.7.5.)

- **Chunked matmul for large eval workloads.** Phase 2-or-later,
  performance. The 2.7.5 numpy implementation in
  `SemanticRanker.search()` allocates the full
  `(n_corpus, n_queries) × float32` similarity matrix in memory.
  Current eval workloads are well under the threshold:
  nl2bash 10,624 × 11,348 × 4 = **482 MB peak**; dogfood ≈ 1 KB;
  open-source corpus (issue #13, planned ~1000 corpus × 100
  queries) ≈ 400 KB. **Trigger condition: revisit when any single
  eval workload approaches 1 GB peak memory budget.**

  Chunking strategy when needed: compute
  `corpus_emb @ query_emb_chunk.T` for chunks of ~1024 queries at
  a time (≈ 40 MB intermediate). Trivially implementable; just
  adds a Python loop around the existing matmul. No algorithmic
  change. Tag: tech-debt, performance, phase-2-or-later.

- **`HARD_RUNTIME_FAIL_S` default should be context-aware, not
  hardcoded in `run_eval()`.** Phase 2 cleanup. The current default
  of 120s baked an execution-context assumption (local M-series
  development) into source code that is now called from multiple
  contexts (CI Linux needs 360s; the behavior-preservation test
  added 2.7 was the first new caller to expose this). The 2.7
  workflow patches the symptom by setting the env override on the
  pytest step too; the deeper fix is to make the default
  context-aware (e.g. detect CI via env var, scale by available
  cores, or have the harness sample one rapid eval and infer a
  budget). Tag: tech-debt, ci, eval-quality.
- **2.9 hybrid search (vector + FTS5 with RRF k=60)** — deferred from
  v1 to v1.1. Original Phase 2 plan included this commit before Phase
  3; the 2.8 closure decision was to defer based on (a) 4× dogfood and
  1.11× nl2bash semantic-only deltas suggesting hybrid is marginal-
  gain, (b) Phase 3 producing actual user-feedback data that should
  inform whether hybrid is worth shipping, (c) v1 launch story not
  requiring it. v1.1 work begins after launch with real user query
  data informing the implementation. Tag: phase-2-deferred,
  post-launch.
- **Document first-call latency in README** — Phase 4 README content.
  The MCP server lazy-loads the embedder on first tool call (~5s for
  model load + MPS warmup); subsequent calls are fast. Document this
  under "Expected behavior on first use" in the README's getting-
  started section so users don't mistake the first-call delay for a
  hang. Tag: documentation, pre-launch.
- **Revisit `@server.list_tools()` type-ignore in `src/recall/server.py`
  when mcp ships full type stubs (likely v1.x).** Currently necessary
  due to SDK 1.27 stub incompleteness — the decorator-based
  handler-registration pattern returns Any, which mypy strict mode
  flags as no-untyped-call / untyped-decorator. The pattern is correct
  SDK usage; the type-ignore is targeted (`no-untyped-call,
  untyped-decorator`) and limited to the single decorator line. When
  the SDK ships type stubs, drop the ignore and the corresponding
  ``[[tool.mypy.overrides]]`` block in ``pyproject.toml``. Not blocking.
  Tag: tech-debt, low-priority.
- **Real-history scrubber validation as a v1 README claim** —
  Phase 4 README content. The 2026-05-04 dogfood-prep audit caught
  53 real secrets (URL userinfo + JWTs) across 1800 lines of real
  shell history with zero misses. This supports a "validated on
  real shell history" claim in the v1 README that's stronger than
  any quantity of synthetic-test evidence — the synthetic corpus
  proves regression prevention, the real-history audit proves
  real-world coverage. Land in the README's privacy / trust
  section at v1 launch. Tag: documentation, phase-4.
- **NL paraphrase-distance validator for the eval harness** —
  Phase-2-or-later, eval-quality. Automated shared-token analysis
  between NL and command after FTS5 tokenization, output a
  paraphrase-distance score (0 = full overlap, 1 = zero shared
  tokens). Would gate any new query landing in `eval/dogfood.toml`
  (or future expansion sets) at the tagged tier — L permits
  overlap, M requires a minimum distance threshold, H requires
  near-zero overlap. ~30 lines of Python. Surfaced by the 2.6.5
  dogfood-prep round: 3 of 5 first-draft NLs were vocabulary-
  overlapping at the L tier when intended as M / H, caught only
  by manual re-audit. Validator would prevent the whole class of
  issue at scale. Tag: tech-debt, eval-quality, phase-2-or-later.

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

## Commit attribution

Commits in this repo are authored solely by the project owner. Do NOT add
`Co-Authored-By: Claude`, `Generated with Claude Code`, `🤖`,
`noreply@anthropic.com`, or any other AI-attribution trailer to commit
messages, PR descriptions, or issue bodies. The git author config
(`Tarun Sharma <tarun.sharma@ieee.org>`) is the sole attribution. This
applies to all future commits without exception. Default Claude Code
commit-template behavior includes such a trailer — explicitly omit it.

**Squash-merge workflow.** Do NOT use `gh pr merge --squash` — GitHub's
squash-merge uses the GitHub-account noreply identity
(`<id>+<user>@users.noreply.github.com`) for the squash commit author,
not the local git config. This was caught at 2.7 and fixed by amending +
force-pushing; named here so future PR merges don't repeat it.

Instead, squash-merge locally:

```bash
git checkout main && git pull --ff-only origin main
git merge --squash verify/<branch>
git commit -F /tmp/<commit-msg-file>   # explicit body; NO trailers
git push origin main
git push origin --delete verify/<branch>
```

This uses the local git config's author (the rule above) and gives full
control over the commit message via `-F`. The `gh pr <N>` PR will
auto-close on the GitHub side because the squash commit's contents
match the merged PR's diff.
