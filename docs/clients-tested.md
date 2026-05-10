# clients-tested.md

> **v1 launch status:** 1/6 entries filled with real observations
> (Claude.ai web MCP connector — the bug-surfacing surface that
> produced findings F1-F5 below). Per the locked launch-readiness
> contract (CLAUDE.md "v1 launch-readiness gate (3.13)"), 4 macOS
> desktop/IDE clients + 1 Windows entry are explicitly `not yet
> verified` — manual UX verification is human-eyeball work the v1
> launch sprint isn't budgeting time for. Tracked as v1.0.1
> deliverable: "macOS desktop client matrix expansion."
>
> The Claude.ai web entry is the load-bearing one: it surfaced the
> findings the v1 README's trust signal depends on, and the F1-F5
> evidence trail in this file documents recall's posture toward
> shipping bugs honestly with explicit fix paths.

Per-client verification matrix for `recall-mcp`. Each entry captures:
client+OS+recall versions, date verified, the 7-item assertion-
checklist outcome from [`tests/manual_smoke.md`](../tests/manual_smoke.md)
Section C, free-text observations (including natural-language summary
quality from the client's LLM), and any known issues.

This file is the *evidence* file. The *procedure* lives at
[`tests/manual_smoke.md`](../tests/manual_smoke.md) Section C —
HOW to verify; this file records WHAT was observed.

The README's "Tested clients" table (added in Phase 4) quotes from
this file directly.

## Glossary

| symbol  | meaning                                                                |
| ------- | ---------------------------------------------------------------------- |
| pass    | assertion held in the maintainer's session                             |
| skip    | not applicable to this client (e.g. tool palette UI absent)            |
| fail    | assertion did not hold; details in the entry's "Known Issues" section  |
| —       | not yet verified (entry pending; see launch-readiness contract below)  |

## Cadence

Re-verify each entry when:

1. **Major recall version change** (v1.0 → v1.1 → v2.0)
2. **Recall PR touches user-facing surface** — tool descriptions,
   output structure, error messages. Reviewer flags the PR as
   needing client re-verification.
3. **Client major version update** (e.g. Claude Desktop 1 → 2). Maintainer
   monitors manually for v1; automated tracking is a v1.x deferred item.

## v1 launch-readiness contract

Before v1 announce:

- All entries below filled with **real observations** — not placeholder
  text. If an entry hasn't been verified yet, leave it as `—` (not
  yet verified). Shipping `1/4 honestly` with a README caveat is
  preferred over `4/4 carelessly` (CLAUDE.md "v1 launch-readiness
  gate (3.13)").
- `stdio-windows` CI lane green on main.
- All Phase 3 commits (3.9-3.13) on main with green CI.

---

## Pending fixes (3.13.5)

Surfaced during the Claude.ai MCP connector verification session that
exercised real tools/call dispatch against an indexed corpus. These
findings predate 3.13 (introduced in 3.10's tool implementations);
3.13's matrix surfaced them. Both ship as a 3.13.5 hotfix on a fresh
branch after 3.13 squash-merges.

### F1 — `commands_after` returns empty `following` for non-atuin indexes

**Symptom:** When the index has only zsh/bash entries (no atuin),
`commands_after` returns `pattern_match` correctly but `following`
is always `[]`. The tool *runs successfully* (no error) but provides
no sequence data — silent UX degradation.

**Root cause:** zsh/bash readers don't capture `session_id`. The
SQL lookup `WHERE session_id = ? AND ts > ...` returns no rows when
`session_id IS NULL`. Handler explicitly skips the lookup in that
case (returns empty `following`). Behavior is consistent with the
description's "in the same session" wording but the description
under-emphasizes the atuin dependency.

**Fix path (3.13.5):** Make `commands_after` atuin-required like
`failed_recently`. Add `state: no_atuin_source` check; return clean
state-error with this message:

> commands_after requires atuin source data for session-boundary
> tracking. Your index doesn't include atuin records. Install atuin
> (https://atuin.sh) to capture session ids for new commands, then
> run 'recall index' to incrementally pick them up. (Without atuin,
> sequence tracking would conflate commands across concurrent
> terminal sessions.)

The parenthetical surfaces the correctness reason — closes the
"couldn't you just guess?" question for the LLM client (and through
it the user) before it gets asked. Cross-session conflation is real:
two terminals running independently produce ts-adjacent commands
with no causal relationship.

**Why not a timestamp-adjacency heuristic?** Considered (option (b)
in 3.13's diagnosis); rejected. Single-terminal users would get
useful results; multi-terminal users would get occasional incorrect
causality. A semantic-search tool's value depends on result
trustworthiness; conflating sessions corrupts that.

**Cost:** ~10 LoC + 1 fixture update + tool description tweak.

### F2 — `find_in_project` not appearing in client tool palette (5/6 visible)

**Symptom:** Out of the 6 registered tools, 5 appear in the Claude.ai
MCP connector's tool palette: `search`, `commands_after`,
`failed_recently`, `command_stats`, `recent`. **`find_in_project`
specifically is missing.** Original 3.13 framing of this finding
("only 4/6 visible") was wrong; updated observation has 5/6 visible
with `find_in_project` as the lone holdout.

**Earlier hypothesis refuted.** The "missing required-key" hypothesis
applied to tools with no required parameters; `find_in_project` HAS
a required parameter (`query`), so that hypothesis can't explain why
it specifically is filtered.

**Open hypothesis space (none confirmed):**
1. Client treats `find_in_project` as a duplicate/subset of `search`
   (overlapping semantic surface; both take `query` + an optional
   path filter) and dedups one.
2. Description content trips a content filter (mentions environment
   variable `MCP_CLIENT_CWD`, mentions filesystem paths).
3. Position 2 in TOOLS list interacts with a client UI heuristic.

**Verification needed (3.13.5 plan):**
1. Capture actual `tools/list` JSON the Claude.ai MCP connector
   received; compare to 3.10 manual smoke baseline schema for
   `find_in_project`.
2. Test the same recall server against one other MCP client (Cursor
   or Zed). If `find_in_project` appears there → Claude.ai-connector-
   specific filtering. If it doesn't appear in any client → some
   shape of our tool registration is incompatible.

**Fix path (3.13.5) — baseline + investigation:**

Apply the original 3-line `setdefault("required", [])` shim in
`_tool_for()` regardless. Doesn't change semantics; makes schemas
explicit; helps any client that filters on missing-required-key:

```python
def _tool_for(name, description, model):
    schema = model.model_json_schema()
    schema.setdefault("required", [])  # MCP-client compatibility
    return Tool(name=name, description=description, inputSchema=schema)
```

**This shim is unlikely to be the load-bearing fix for find_in_project
specifically** (its schema already has `"required": ["query"]`). The
investigation step above identifies the real cause; the fix shape
adjusts based on outcome (description rewording, schema tweak, or
filing as upstream client bug).

**Cost:** ~3 LoC for the shim + ~30 min investigation effort + fix
shape TBD. Probably ~10-30 LoC total.

### F4 — `recent` returns wrong-order data on zsh-only indexes

**Symptom:** `recent` (the canonical "show me my most recent
commands" tool) returns OLDEST-first commands when the index is
zsh-only. Tool description claims "Time-ordered DESC, id ASC
tie-break"; actual behavior contradicts the description silently.

**Root cause:** zsh's `EXTENDED_HISTORY` flag is not always set;
when absent, the zsh source emits entries with `ts = 0` (unknown
timestamp, per HistorySource protocol). With `ts` uniformly 0
across the result set, the tie-break (`id ASC`) dominates →
oldest-insertion-first. Same silent-degradation shape as F1
(commands_after): tool succeeds, returns data, but data doesn't
match documented semantics.

**Fix path (3.13.5):** Hybrid ORDER BY in `_handle_recent`. When
`ts` is known (>0), use it; when unknown (=0), fall back to
insertion order (`id DESC`) as a time-proxy:

```sql
ORDER BY
  CASE WHEN ts > 0 THEN 0 ELSE 1 END,  -- real-ts rows first
  ts DESC,
  id DESC                              -- fallback: most-recent insertion first
```

This treats `ts = 0` as "unknown timestamp; use insertion order"
rather than as "literally epoch-zero." Mixed-source indexes
(atuin + zsh, where atuin has real `ts` and zsh has 0) get
atuin rows first by ts, then zsh rows by id-descending — both
groups in most-recent-first order.

**Why not atuin-required (parallel to F1)?** `recent` is the most
basic tool — "show me my history." Forcing atuin for it is too
aggressive for v1. `id DESC` is a correctness-preserving fallback
(insertion order is ~time order for shell history) without the
cross-session-conflation risk that ruled out the heuristic for
`commands_after` (where the question is causal: "what came after
X").

**Same fix applies anywhere we order by `ts`.** Other handlers
(`commands_after` post-F1-fix, `failed_recently`) can use the
same pattern; v1 only `recent` needs it for non-atuin users.

**Cost:** ~5 LoC (SQL change) + 1-2 fixture updates + tool
description tweak.

### F5 — `failed_recently` 4+ minute hang (observed once, monitoring)

**Symptom (single observation):** During a Claude.ai MCP connector
session that had previously made multiple semantic search calls
against an indexed corpus, `failed_recently` against a non-atuin
index hung for 4+ minutes. Claude.ai's connector emitted a "MCP
server may be crashed" timeout warning. Re-running the same call
against the same index returned the clean state-error in ~8ms.

**Original hypothesis (asyncio dispatch serialization) — REFUTED.**
Diagnostic spike (2026-05-10) confirmed: SDK dispatch is concurrent.
Fired `failed_recently` 50ms after a cold-cache `search` (24-second
model load); `failed_recently` returned a clean state-error in 8ms
while `search` was still 24 seconds away from completing. Recall log
captured both handler dispatches in the same wall-clock second:

```
09:58:06 INFO recall.server: lazy-loading embedder (first tool call)
09:58:06 INFO recall.server.tools tool=failed_recently state_error count=0
09:58:30 INFO recall.server: embedder ready (model=BAAI/bge-small-en-v1.5, dim=384)
09:58:30 INFO recall.server.tools tool=search ok count=3
```

Asyncio scheduling correctly interleaves handlers; the embedder
lock doesn't block unrelated tool calls; the SDK does NOT serialize
at the dispatch layer (despite serializing in the simple two-fast-
handler case from the 3.12 spike).

**Hypothesis space remaining (none confirmed):**
- Claude.ai MCP connector-specific behavior under load (transport
  framing, request batching, network-side retry)
- Transient OS-level pipe stall
- Network/transport-level glitch specific to that session

**Status: monitoring. No code fix to ship.** Without reproduction,
there's no fix to verify. Cannot ship a fix against an unreproducible
bug — would have no way to know if the fix worked.

**User feedback channel for v1.0.1:** if this recurs, we want
diagnostic data to investigate. **If you encounter this:**

1. File a GitHub issue with the contents of `~/.recall/logs/recall.log`
   from the affected session (timestamp window around the hang).
2. If you can reproduce it: `py-spy dump --pid $(pgrep -f 'recall serve')`
   while the hang is in progress and attach the stack trace.
3. Note the client (Claude.ai web, Claude Desktop, Cursor, etc.) and
   what tool calls preceded the hang in the same session.

This converts F5 from launch-blocker into a v1.0.1 user-feedback
channel.

Each client's "Known Issues" subsection below cross-references this
"Pending fixes (3.13.5)" section rather than restating the findings.
After 3.13.5 lands, entries here update from "Pending fix" to
"Fixed in 3.13.5" with the squash-commit SHA — preserves the
evidence trail.

### Finding 3 — README framing (Phase 4, not 3.13.5)

Most metadata fields (cwd, hostname, ts, session_id) are null for
zsh/bash entries because those readers don't capture them. Not a
bug; an honest limitation. Phase 4 README needs framing as value-
add: "atuin integration enables richest UX; without it, semantic
search + frequency analytics still work fully." Familiar
"install X for full experience" pattern. Tracked as Phase 4
deferred entry in CLAUDE.md.

---

## Claude.ai web (MCP connector) — first filled-in entry

| field           | value |
| --------------- | ----- |
| Client          | Claude.ai web app, MCP connector surface |
| Client version  | (web app — version N/A; verified against then-current production) |
| OS              | N/A (browser-based; Claude.ai's connector is the MCP host) |
| Recall version  | verify/3.13 @ commit `43d652b` (3.13 + evidence trail) |
| Date verified   | 2026-05-09 — 2026-05-10 (multi-session verification surfaced F1, F2, F4, F5) |

**Important framing**: this entry uses Claude.ai's MCP connector — a
web-app surface, not a desktop tool palette. Verification done via
AI assistant intermediating tool calls; **rendering and natural-
language summary observations are N/A for this client surface** — the
MCP connector returns structured data to Claude.ai, which the
assistant LLM may further summarize, but there's no end-user "tool
palette" UI to evaluate the way Claude Desktop, Cursor, Zed, or Cline
have.

This entry's value is its role as the **bug-surfacing surface**. The
verification session against this client is what produced findings
F1-F5 listed in "Pending fixes (3.13.5)" above. Subsequent macOS
clients (Cursor, Claude Desktop, Zed, Cline) verify the UX dimensions
that don't apply here.

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools                                                 | **fail** — 5/6 visible; `find_in_project` missing (see F2) |
| 2 | tool descriptions render fully                                                 | pass (visible tools' descriptions rendered fully in the assistant's tool-discovery context) |
| 3 | tools/call recent succeeds + renders readably                                  | **fail** — succeeds but returns wrong-order data on zsh-only index (see F4) |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | pass — clean validation error text returned in milliseconds |
| 5 | tools/call search triggers embedder load; client shows progress                | pass — verified via 2026-05-10 spike: 24s cold-cache load, no frozen state |
| 6 | no tracebacks visible to user                                                  | pass |
| 7 | recall.log shows structured `tool=<name>` lines                                | pass — verified via `tail -f ~/.recall/logs/recall.log \| grep '"tool":'` |

### Observations

The verification value of this client surface is bug-surfacing, not
UX evaluation. AI-assistant-mediated tool dispatch exposes correctness
issues (data shape, response timing, error rendering) that don't
require visual UI. Findings F1-F5 above were all surfaced through
this surface; tooling for client-specific UX (palette rendering,
natural-language summary quality, tool-call discoverability in the
chat flow) is the natural complement run from the macOS desktop +
IDE clients below.

One additional observation worth noting (not a bug, observable in
this surface): when assistant reasoning uses `commands_after` against
a zsh-only index and gets empty `following`, the LLM correctly
recognizes the absence of useful sequence data and either suggests
running `recall index --source atuin` or moves to a different tool —
suggesting the LLM has reasonable failure recovery behavior even
with the current silent-degradation bug. F1's fix improves this
further by making the limitation explicit at the protocol layer.

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section. F1-F5 were all
  surfaced from this client surface during the 2026-05-09 — 2026-05-10
  verification sessions.

---

## Cursor (macOS)

| field           | value |
| --------------- | ----- |
| Client version  | _not yet verified_ |
| OS version      | _not yet verified_ |
| Recall version  | 0.0.1 |
| Date verified   | _not yet verified_ |

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools in palette                                      | —      |
| 2 | tool descriptions render fully (or noted truncation)                           | —      |
| 3 | tools/call recent succeeds + renders readably                                  | —      |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | —      |
| 5 | tools/call search triggers embedder load; client shows progress (no frozen UI) | —      |
| 6 | no tracebacks visible to user                                                  | —      |
| 7 | recall.log shows structured `tool=<name>` lines                                | —      |

### Observations

_(maintainer fills in — natural-language summary quality from the
client's LLM, rendering quirks, latency notes, anything notably good
or notably bad)_

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section for server-side
  findings (F1: commands_after non-atuin; F2: only-4-of-6-tools). Both
  apply to any client and ship as a 3.13.5 hotfix.
- _(client-specific quirks: maintainer fills as observations accumulate)_

---

## Claude Desktop (macOS)

| field           | value |
| --------------- | ----- |
| Client version  | _not yet verified_ |
| OS version      | _not yet verified_ |
| Recall version  | 0.0.1 |
| Date verified   | _not yet verified_ |

> Pre-3.13 reference: a partial Claude Desktop verification was run
> at the 3.10 squash-merge gate. Result: tools/list rendered all 6
> tools; the `recent` tool's no-index error rendered as natural-
> language guidance; refresh transition (3.10 follow-up) verified
> end-to-end. That session predates the canonical 7-item checklist;
> formalized re-verification fills the entry below.

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools in palette                                      | —      |
| 2 | tool descriptions render fully (or noted truncation)                           | —      |
| 3 | tools/call recent succeeds + renders readably                                  | —      |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | —      |
| 5 | tools/call search triggers embedder load; client shows progress (no frozen UI) | —      |
| 6 | no tracebacks visible to user                                                  | —      |
| 7 | recall.log shows structured `tool=<name>` lines                                | —      |

### Observations

_(maintainer fills in — natural-language summary quality from the
client's LLM, rendering quirks, latency notes, anything notably good
or notably bad)_

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section for server-side
  findings (F1: commands_after non-atuin; F2: only-4-of-6-tools). Both
  apply to any client and ship as a 3.13.5 hotfix.
- _(client-specific quirks: maintainer fills as observations accumulate)_

---

## Zed (macOS)

| field           | value |
| --------------- | ----- |
| Client version  | _not yet verified_ |
| OS version      | _not yet verified_ |
| Recall version  | 0.0.1 |
| Date verified   | _not yet verified_ |

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools in palette                                      | —      |
| 2 | tool descriptions render fully (or noted truncation)                           | —      |
| 3 | tools/call recent succeeds + renders readably                                  | —      |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | —      |
| 5 | tools/call search triggers embedder load; client shows progress (no frozen UI) | —      |
| 6 | no tracebacks visible to user                                                  | —      |
| 7 | recall.log shows structured `tool=<name>` lines                                | —      |

### Observations

_(maintainer fills in)_

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section for server-side
  findings (F1: commands_after non-atuin; F2: only-4-of-6-tools). Both
  apply to any client and ship as a 3.13.5 hotfix.
- _(client-specific quirks: maintainer fills as observations accumulate)_

---

## Cline VS Code (macOS)

| field           | value |
| --------------- | ----- |
| Client version  | _not yet verified_ |
| OS version      | _not yet verified_ |
| Recall version  | 0.0.1 |
| Date verified   | _not yet verified_ |

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools in palette                                      | —      |
| 2 | tool descriptions render fully (or noted truncation)                           | —      |
| 3 | tools/call recent succeeds + renders readably                                  | —      |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | —      |
| 5 | tools/call search triggers embedder load; client shows progress (no frozen UI) | —      |
| 6 | no tracebacks visible to user                                                  | —      |
| 7 | recall.log shows structured `tool=<name>` lines                                | —      |

### Observations

_(maintainer fills in)_

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section for server-side
  findings (F1: commands_after non-atuin; F2: only-4-of-6-tools). Both
  apply to any client and ship as a 3.13.5 hotfix.
- _(client-specific quirks: maintainer fills as observations accumulate)_

---

## Claude Desktop (Windows 11 ARM, UTM VM)

| field           | value |
| --------------- | ----- |
| Client version  | _not yet verified — pre-launch UTM VM session_ |
| OS version      | _Windows 11 ARM (UTM VM on Apple Silicon)_ |
| Recall version  | 0.0.1 |
| Date verified   | _not yet verified_ |

> Setup notes for the maintainer: install [UTM](https://mac.getutm.app/)
> (free, Apple Silicon native), download Microsoft's free [Windows 11
> ARM development VM](https://developer.microsoft.com/en-us/windows/downloads/virtual-machines/),
> install Claude Desktop for Windows, install recall via `pip install -e .`
> from the cloned repo (uvx flow may not be available out-of-the-box).
>
> Permanent CI gate for Windows stdio-cleanliness lives in the
> `stdio-windows` CI lane (3.13). This manual entry covers the UX
> dimensions CI can't reach: Claude Desktop's Windows-specific
> rendering, tool palette behavior on Windows, MCP config file
> location (`%APPDATA%\Claude\claude_desktop_config.json`).

### Assertion checklist

| # | check                                                                          | result |
| - | ------------------------------------------------------------------------------ | ------ |
| 1 | tools/list renders all 6 tools in palette                                      | —      |
| 2 | tool descriptions render fully (or noted truncation)                           | —      |
| 3 | tools/call recent succeeds + renders readably                                  | —      |
| 4 | tools/call command_stats `pattern: "%"` returns clean user-facing error       | —      |
| 5 | tools/call search triggers embedder load; client shows progress (no frozen UI) | —      |
| 6 | no tracebacks visible to user                                                  | —      |
| 7 | recall.log shows structured `tool=<name>` lines                                | —      |

### Observations

_(maintainer fills in pre-launch via UTM VM session)_

### Known issues

- See top-level **"Pending fixes (3.13.5)"** section for server-side
  findings (F1: commands_after non-atuin; F2: only-4-of-6-tools). Both
  apply to any client and ship as a 3.13.5 hotfix.
- _(client-specific quirks: maintainer fills as observations accumulate)_
