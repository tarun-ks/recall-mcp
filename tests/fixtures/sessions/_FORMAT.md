# Recorded session fixture format (3.12)

This directory holds end-to-end MCP protocol scenarios as JSONL files.
The replay engine in `tests/test_recorded_sessions.py` walks each
fixture line-by-line, sending requests, reading responses, and
matching observed responses against expected ones with sentinel-
aware comparison.

## File layout

Each fixture is a single `.jsonl` file. One frame per line.
Lines starting with `#` are comments OR structural directives.

```
# Scenario: <one-line description>.
# DB state: <required state>.
# Lane runtime contribution: <approx>.

> initialize {"protocolVersion":"2024-11-05",...}
< {"jsonrpc":"2.0","id":<INT>,"result":{...}}
> notifications/initialized {}
> tools/call {"name":"recent","arguments":{"limit":3}}
< {"jsonrpc":"2.0","id":<INT>,"result":{...}}
```

## Direction markers

- `>` — client → server. Format: `> <method> <params-json>`.
  Notifications use the same syntax; the loader recognizes
  `notifications/*` methods and skips the response-read step.
- `<` — server → client (expected response). Format: `< <full-json>`.

## Comments

Any line starting with `#` is a comment. Comments inside a frame
position (between `>` lines) are skipped by the loader. The first
few lines of each file SHOULD document scenario purpose, required
DB state, and approximate lane runtime contribution.

## Sentinels

Inside `<` frames, certain tokens are sentinels matched loosely:

| sentinel  | matches                                       |
| --------- | --------------------------------------------- |
| `<TS>`    | any positive integer (unix-second timestamps) |
| `<HEX16>` | a 16-char hex string `[0-9a-f]{16}`           |
| `<SCORE>` | a float in `[-1.0, 1.0]` (cosine similarity)  |
| `<INT>`   | any integer                                   |
| `<ANY>`   | any JSON value (escape hatch — see below)     |

Anywhere else, exact equality is enforced.

**`<ANY>` is the escape hatch.** When a fixture has more than 2
`<ANY>` sentinels, the replay engine emits a pytest warning visible
in CI logs ("escape-hatch erosion"). It does NOT fail the build —
the warning surfaces drift toward over-loose fixtures so the
maintainer can tighten them. Hard cap is intentionally not enforced;
warning at the right cadence is the discipline.

## Structural directives

Lines beginning with these tokens are parser directives, not frames:

### `#PIPELINE-START` / `#PIPELINE-END`

Marks a pipelined block. Inside the block:

```
#PIPELINE-START
> tools/call {"name":"recent","arguments":{"limit":3}}      # tag:recent
> tools/call {"name":"command_stats","arguments":{"pattern":"git"}}  # tag:stats
< {"jsonrpc":"2.0","id":<INT>,"result":{...recent shape...}}  # tag:recent
< {"jsonrpc":"2.0","id":<INT>,"result":{...stats shape...}}   # tag:stats
#PIPELINE-END
```

The replay engine:
1. Sends every `>` frame in the block back-to-back, capturing each
   request id by its `# tag:<name>` annotation.
2. Reads N responses (where N = count of `<` frames in the block).
3. Builds an id-keyed dict of `{id: response}`.
4. For each expected `<` frame, looks up the request id by tag,
   then matches the actual response (from the dict) against the
   expected JSON.

This means **response order is NOT asserted** — id-correctness is.
That's the protocol contract; in-order arrival is an SDK
implementation detail.

**Tag uniqueness within a block is a format constraint.** Within
a single `#PIPELINE-START` / `#PIPELINE-END` block, each tag must
appear exactly once on a `>` line and exactly once on a `<` line.
Duplicate tags within a block (e.g. from a copy-paste error) are
a `FixtureFormatError`. The loader rejects malformed fixtures with
a clear file:line message.

### `#IDLE <seconds>`

Insert an `await asyncio.sleep(<seconds>)` before the next frame.
Used by the long-idle scenario (S1) — `#IDLE 30` sleeps 30s before
the next tool call to verify the SDK stdio handler survives idle.

## Adding a new fixture

1. Create `tests/fixtures/sessions/<scenario_name>.jsonl`.
2. Write a comment header: scenario purpose, required DB state,
   lane runtime contribution.
3. Add a corresponding `test_replay_<scenario_name>` function in
   `tests/test_recorded_sessions.py`. Each scenario gets its own
   function so test reports name the failing scenario clearly.
4. Run `pytest tests/test_recorded_sessions.py -v` locally.

## Fixture drift policy

When the mcp SDK or recall's tool surface changes intentionally,
fixtures need updating in the same commit as the surface change.
**Reviewer must verify fixture changes match the surface change
line-by-line — not rubber-stamp.**

CI surfaces fixture-drift PRs via the `fixture drift warning` lane:
when a PR touches both `tests/fixtures/sessions/*.jsonl` AND
`src/recall/{server,tools}.py`, a `::warning::` annotation appears
on the lane summary asking the reviewer to check the fixture diff
against the surface change.

If fixture count grows past ~10, revisit with a record-mode tool
that captures fixtures from a live session (deferred-items entry).
