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

- (none yet — fill bullet list as observations accumulate)

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

- (none yet)

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

- (none yet)

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

- (none yet)

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

- (none yet)
