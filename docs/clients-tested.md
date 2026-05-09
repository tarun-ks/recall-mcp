# clients-tested.md

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

### F2 — only 4 of 6 tools registered in MCP client

**Symptom:** `recent` and `failed_recently` not appearing in the
client's tool palette. Other 4 tools (search, find_in_project,
commands_after, command_stats) appear normally.

**Hypothesis:** The 4 visible tools all have at least one required
parameter (`query` or `pattern`); the 2 missing tools have no
required parameters (all fields default or nullable). Pydantic v2's
`model_json_schema()` **omits the `"required"` key entirely when
no fields are required** (vs emitting `"required": []`). Some MCP
clients filter or hide tools whose schema lacks a `required`
declaration.

**Verification needed before fix (3.13.5 plan should include):**
1. Capture actual tools/list JSON from a fresh Claude Desktop session
   against current main; compare to 3.10 manual smoke baseline.
2. Test the same recall server against one other MCP client (Cursor
   or Zed). If `recent`/`failed_recently` appear there → Claude.ai
   MCP connector-specific filtering, fix framed as "compatibility
   shim." If they don't appear in any client → JSON Schema convention
   issue, fix framed as "spec compliance."

**Fix path (3.13.5) — same regardless of root cause:**
3-line `setdefault("required", [])` in `_tool_for()`:

```python
def _tool_for(name, description, model):
    schema = model.model_json_schema()
    schema.setdefault("required", [])  # MCP-client compatibility
    return Tool(name=name, description=description, inputSchema=schema)
```

Doesn't change semantics; makes the schema explicit. Backwards-
compatible with clients that already work.

### Per-client cross-reference

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
