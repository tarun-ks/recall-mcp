"""Recorded MCP session replay tests (Commit 3.12).

Companion to ``tests/test_stdio_cleanliness.py`` (3.11). Where 3.11
tests the BYTES are clean (transport invariant), 3.12 tests the
PROTOCOL behaves correctly across multi-frame scenarios that single-
call subprocess tests can't reach.

Five scenarios (per the locked Phase 3.12 §4 + refinement):
  S1 long_idle_then_call  — 30s idle then tool call (marquee)
  S2 happy_path           — initialize → list → call (positive case)
  S3 error_then_recovery  — error doesn't wedge subsequent calls
  S4 multiple_back_to_back — sequential calls preserve state
  S6 pipelined_requests   — id-correctness under pipelined sends

S5 (state-error then refresh) is NOT included; redundant with 3.11
test E (covered by lazy-refresh subprocess test).

Format: tests/fixtures/sessions/*.jsonl. Spec lives at
tests/fixtures/sessions/_FORMAT.md.

Lane: extends the existing ``stdio (ubuntu / py3.12)`` CI lane (3.11).
3.12 adds ~35s lane runtime (S1's 30s idle + ~5s for S2-S6).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pytest

# Sentinel tokens. Documented in tests/fixtures/sessions/_FORMAT.md.
_SENTINEL_TS = "<TS>"
_SENTINEL_HEX16 = "<HEX16>"
_SENTINEL_SCORE = "<SCORE>"
_SENTINEL_INT = "<INT>"
_SENTINEL_ANY = "<ANY>"
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_TAG_RE = re.compile(r"#\s*tag:([A-Za-z0-9_-]+)")

SESSIONS_DIR = Path(__file__).parent / "fixtures" / "sessions"

_ANY_WARN_THRESHOLD = 2  # warn if a fixture uses <ANY> more than this


# === Loader ===


class FixtureFormatError(ValueError):
    """Raised on malformed fixture files. Includes file path + line number
    + reason in the message."""


@dataclass(frozen=True)
class FixtureFrame:
    """One frame in a recorded session fixture."""

    direction: Literal[">", "<"]
    method: str | None  # set for > frames; None for < frames
    payload: dict[str, Any]  # params for >; full envelope for <
    tag: str | None  # from "# tag:X" trailing comment, if any
    line_no: int  # for error messages

    @property
    def is_notification(self) -> bool:
        return (
            self.direction == ">"
            and self.method is not None
            and self.method.startswith("notifications/")
        )


@dataclass
class FixtureScenario:
    """Parsed scenario: ordered frames + structural directives."""

    path: Path
    frames: list[FixtureFrame]
    pipeline_blocks: list[tuple[int, int]]  # half-open [start_idx, end_idx)
    idle_at: dict[int, int]  # frame_idx → seconds to sleep BEFORE that frame
    any_count: int = field(default=0)


def _strip_inline_comment(s: str) -> tuple[str, str | None]:
    """Strip an inline ``# tag:X`` annotation; return (head, tag-or-None).

    Only strips ``# tag:NAME`` patterns; other ``#`` content is left intact
    so JSON containing # in strings doesn't get mangled.
    """
    m = _TAG_RE.search(s)
    if m:
        return s[: m.start()].rstrip(), m.group(1)
    return s, None


def _parse_request_line(content: str, line_no: int, path: Path) -> tuple[str, dict[str, Any]]:
    """Parse `> <method> <params-json>` (params-json may be empty)."""
    parts = content.split(" ", 1)
    method = parts[0]
    if len(parts) == 1:
        params: dict[str, Any] = {}
    else:
        try:
            params = json.loads(parts[1])
        except json.JSONDecodeError as e:
            raise FixtureFormatError(
                f"{path}:{line_no}: invalid JSON params for method {method!r}: {e}"
            ) from e
    return method, params


def _parse_response_line(content: str, line_no: int, path: Path) -> dict[str, Any]:
    """Parse `< <full-json>`."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        raise FixtureFormatError(f"{path}:{line_no}: invalid JSON response: {e}") from e
    return payload


def load_scenario(path: Path) -> FixtureScenario:
    """Parse a JSONL fixture file. Raises FixtureFormatError on malformed input.

    Fixture validation enforced here:
      - Tag uniqueness within a pipeline block (one > and one < per tag)
      - #PIPELINE-START / #PIPELINE-END must be balanced
      - #IDLE <int> must precede a frame
    """
    if not path.exists():
        raise FileNotFoundError(f"fixture not found: {path}")

    frames: list[FixtureFrame] = []
    pipeline_blocks: list[tuple[int, int]] = []
    idle_at: dict[int, int] = {}
    any_count = 0

    in_pipeline = False
    pipeline_start_idx = -1
    pipeline_request_tags: set[str] = set()
    pipeline_response_tags: set[str] = set()
    pending_idle: int | None = None

    with path.open(encoding="utf-8") as f:
        for raw_line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            stripped = line.strip()

            # Empty line: just resets pending_idle? No — keep pending_idle
            # so blank lines between #IDLE and the frame don't lose it.
            if not stripped:
                continue

            # Structural directives
            if stripped.startswith("#PIPELINE-START"):
                if in_pipeline:
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: nested #PIPELINE-START not supported"
                    )
                in_pipeline = True
                pipeline_start_idx = len(frames)
                pipeline_request_tags = set()
                pipeline_response_tags = set()
                continue

            if stripped.startswith("#PIPELINE-END"):
                if not in_pipeline:
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: #PIPELINE-END without matching START"
                    )
                # Tag uniqueness: each tag appears exactly once on > and once on <
                if pipeline_request_tags != pipeline_response_tags:
                    only_req = pipeline_request_tags - pipeline_response_tags
                    only_resp = pipeline_response_tags - pipeline_request_tags
                    msg = []
                    if only_req:
                        msg.append(f"request tags without responses: {sorted(only_req)}")
                    if only_resp:
                        msg.append(f"response tags without requests: {sorted(only_resp)}")
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: tag mismatch in pipeline block; " + "; ".join(msg)
                    )
                pipeline_blocks.append((pipeline_start_idx, len(frames)))
                in_pipeline = False
                pipeline_start_idx = -1
                pipeline_request_tags = set()
                pipeline_response_tags = set()
                continue

            if stripped.startswith("#IDLE"):
                m = re.match(r"^#IDLE\s+(\d+)\s*$", stripped)
                if not m:
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: malformed #IDLE (expected '#IDLE <seconds>')"
                    )
                if pending_idle is not None:
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: consecutive #IDLE "
                        "directives without intervening frame"
                    )
                pending_idle = int(m.group(1))
                continue

            # Plain comment line — skip
            if stripped.startswith("#"):
                continue

            # Frame line: must start with > or <
            if stripped.startswith(">"):
                content = stripped[1:].strip()
                content, tag = _strip_inline_comment(content)
                if not content:
                    raise FixtureFormatError(f"{path}:{raw_line_no}: empty request line")
                method, params = _parse_request_line(content, raw_line_no, path)
                frame = FixtureFrame(
                    direction=">",
                    method=method,
                    payload=params,
                    tag=tag,
                    line_no=raw_line_no,
                )
                if in_pipeline:
                    if tag is None:
                        raise FixtureFormatError(
                            f"{path}:{raw_line_no}: pipelined request "
                            "requires '# tag:<name>' annotation"
                        )
                    if tag in pipeline_request_tags:
                        raise FixtureFormatError(
                            f"{path}:{raw_line_no}: duplicate tag '{tag}' "
                            "on pipelined request line "
                            "(tag uniqueness within a block is required "
                            "— see _FORMAT.md)"
                        )
                    pipeline_request_tags.add(tag)
                if pending_idle is not None:
                    idle_at[len(frames)] = pending_idle
                    pending_idle = None
                frames.append(frame)
                continue

            if stripped.startswith("<"):
                content = stripped[1:].strip()
                content, tag = _strip_inline_comment(content)
                if not content:
                    raise FixtureFormatError(f"{path}:{raw_line_no}: empty response line")
                payload = _parse_response_line(content, raw_line_no, path)
                # Count <ANY> sentinels for the warning threshold
                any_count += _count_any_sentinels(payload)
                frame = FixtureFrame(
                    direction="<",
                    method=None,
                    payload=payload,
                    tag=tag,
                    line_no=raw_line_no,
                )
                if in_pipeline:
                    if tag is None:
                        raise FixtureFormatError(
                            f"{path}:{raw_line_no}: pipelined response "
                            "requires '# tag:<name>' annotation"
                        )
                    if tag in pipeline_response_tags:
                        raise FixtureFormatError(
                            f"{path}:{raw_line_no}: duplicate tag '{tag}' "
                            "on pipelined response line "
                            "(tag uniqueness within a block is required "
                            "— see _FORMAT.md)"
                        )
                    pipeline_response_tags.add(tag)
                if pending_idle is not None:
                    # #IDLE before a < doesn't make protocol sense — idle is
                    # the gap between frames the CLIENT can control.
                    raise FixtureFormatError(
                        f"{path}:{raw_line_no}: #IDLE directly before "
                        "a response line is not supported"
                    )
                frames.append(frame)
                continue

            raise FixtureFormatError(
                f"{path}:{raw_line_no}: unrecognized line; "
                f"expected '>', '<', '#', or empty: {stripped!r}"
            )

    if in_pipeline:
        raise FixtureFormatError(f"{path}: unclosed #PIPELINE-START at end of file")
    if pending_idle is not None:
        raise FixtureFormatError(f"{path}: trailing #IDLE without subsequent frame")

    return FixtureScenario(
        path=path,
        frames=frames,
        pipeline_blocks=pipeline_blocks,
        idle_at=idle_at,
        any_count=any_count,
    )


def _count_any_sentinels(value: Any) -> int:
    """Recursive count of <ANY> sentinel occurrences inside a JSON tree."""
    if value == _SENTINEL_ANY:
        return 1
    if isinstance(value, dict):
        return sum(_count_any_sentinels(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_any_sentinels(v) for v in value)
    return 0


# === Sentinel matcher ===


@dataclass
class MatchFailure:
    path: str
    expected_subtree: Any
    actual_subtree: Any


def match_value(expected: Any, actual: Any, *, path: str = "$") -> MatchFailure | None:
    """Walk expected/actual in parallel; return None on match, MatchFailure on mismatch.

    Sentinel rules:
      <TS>     → any non-negative int (ts is unix-second; 0 allowed for unknown ts)
      <HEX16>  → 16-char hex string
      <SCORE>  → float (or int) in [-1.0, 1.0]
      <INT>    → any int
      <ANY>    → matches anything
      else     → exact equality
    """
    if isinstance(expected, str):
        if expected == _SENTINEL_ANY:
            return None
        if expected == _SENTINEL_TS:
            if isinstance(actual, int) and actual >= 0:
                return None
            return MatchFailure(path, "<TS> (non-negative int)", actual)
        if expected == _SENTINEL_HEX16:
            if isinstance(actual, str) and _HEX16_RE.match(actual):
                return None
            return MatchFailure(path, "<HEX16> (16-char hex string)", actual)
        if expected == _SENTINEL_SCORE:
            if isinstance(actual, (int, float)) and -1.0 <= actual <= 1.0:
                return None
            return MatchFailure(path, "<SCORE> (float in [-1, 1])", actual)
        if expected == _SENTINEL_INT:
            # Booleans are int subclass; reject explicitly.
            if isinstance(actual, int) and not isinstance(actual, bool):
                return None
            return MatchFailure(path, "<INT> (integer)", actual)

    # Dict: same keys + recursive match per key
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return MatchFailure(path, expected, actual)
        if set(expected.keys()) != set(actual.keys()):
            return MatchFailure(
                path,
                f"keys={sorted(expected.keys())}",
                f"keys={sorted(actual.keys())}",
            )
        for key in expected:
            sub = match_value(expected[key], actual[key], path=f"{path}.{key}")
            if sub is not None:
                return sub
        return None

    # List: same length + recursive match per index
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return MatchFailure(path, expected, actual)
        if len(expected) != len(actual):
            return MatchFailure(
                path,
                f"list of length {len(expected)}",
                f"list of length {len(actual)}",
            )
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            sub = match_value(e, a, path=f"{path}[{i}]")
            if sub is not None:
                return sub
        return None

    # Scalar: exact equality
    if expected == actual:
        return None
    return MatchFailure(path, expected, actual)


def _truncate_json(value: Any, max_chars: int = 400) -> str:
    """Render value as JSON, truncate to max_chars with ellipsis marker."""
    s = json.dumps(value, indent=2, default=str)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n... (truncated)"


def _format_failure(fixture_path: Path, frame: FixtureFrame, failure: MatchFailure) -> str:
    """Side-by-side diff with JSON path + truncated subtrees."""
    return (
        f"\n  fixture:        {fixture_path.name}:{frame.line_no}"
        f"\n  json path:      {failure.path}"
        f"\n  expected:       {_truncate_json(failure.expected_subtree)}"
        f"\n  actual:         {_truncate_json(failure.actual_subtree)}"
    )


# === Replay engine ===


def _block_containing(blocks: list[tuple[int, int]], idx: int) -> tuple[int, int] | None:
    for start, end in blocks:
        if start <= idx < end:
            return (start, end)
    return None


def _frame_iter_by_block(
    scenario: FixtureScenario,
) -> Iterator[tuple[int, FixtureFrame, tuple[int, int] | None]]:
    """Yield (idx, frame, block) where block is None for non-pipelined frames."""
    for i, frame in enumerate(scenario.frames):
        block = _block_containing(scenario.pipeline_blocks, i)
        yield i, frame, block


async def replay(session: Any, scenario: FixtureScenario) -> None:
    """Walk the scenario; assert protocol conformance.

    Non-pipelined frames: a > frame and its expected < frame are paired
    by the outer loop; the engine sends the > and matches the actual
    response against the < (skipping over interleaved notifications).

    Pipelined frames: send all > frames in the block (without awaiting
    individual responses), read N responses, build id-keyed dict, match
    each expected < frame against actual by tag → request-id lookup.
    """
    # Warn (not fail) if <ANY> count > threshold per locked §3 refinement
    if scenario.any_count > _ANY_WARN_THRESHOLD:
        import warnings

        warnings.warn(
            f"escape-hatch erosion: {scenario.path.name} uses {scenario.any_count} "
            f"<ANY> sentinels (threshold {_ANY_WARN_THRESHOLD}). "
            "Tighten fixture by replacing <ANY> with more specific sentinels "
            "(<TS>, <HEX16>, <SCORE>, <INT>) or exact values where possible.",
            stacklevel=2,
        )

    i = 0
    while i < len(scenario.frames):
        # Apply any pending idle for this frame index
        if i in scenario.idle_at:
            await asyncio.sleep(scenario.idle_at[i])

        block = _block_containing(scenario.pipeline_blocks, i)
        if block is not None:
            await _replay_pipeline_block(session, scenario, block)
            i = block[1]  # skip past block end
            continue

        frame = scenario.frames[i]
        if frame.direction == ">":
            consumed = await _replay_normal_request(session, scenario, frame, i)
            i += consumed
            continue

        # frame.direction == "<" outside a pipeline block: this is an
        # orphan because the matching > should have consumed it. Real
        # fixture bug.
        raise FixtureFormatError(
            f"{scenario.path}:{frame.line_no}: orphaned response line (no preceding request frame)"
        )


async def _replay_normal_request(
    session: Any,
    scenario: FixtureScenario,
    request_frame: FixtureFrame,
    request_idx: int,
) -> int:
    """Handle a non-pipelined > frame. Returns the number of frame
    indices consumed (1 for notifications, 2+ for request+response).
    """
    assert request_frame.method is not None

    if request_frame.is_notification:
        await session.send_notification(
            request_frame.method,
            request_frame.payload if request_frame.payload else None,
        )
        # Notification consumes only its own > frame.
        return 1

    actual = await session.send_request(
        request_frame.method,
        request_frame.payload if request_frame.payload else None,
    )

    # Find the next < frame, skipping over any notifications.
    response_idx = request_idx + 1
    while response_idx < len(scenario.frames):
        f = scenario.frames[response_idx]
        if f.direction == "<":
            break
        if f.direction == ">" and not f.is_notification:
            # Another request before we found a response — fixture bug.
            raise FixtureFormatError(
                f"{scenario.path}:{request_frame.line_no}: request without matching response line"
            )
        response_idx += 1
    else:
        raise FixtureFormatError(
            f"{scenario.path}:{request_frame.line_no}: request without matching response line"
        )

    response_frame = scenario.frames[response_idx]
    failure = match_value(response_frame.payload, actual)
    if failure is not None:
        pytest.fail(
            f"replay mismatch at request {request_frame.method!r} "
            f"({scenario.path.name}:{request_frame.line_no})"
            + _format_failure(scenario.path, response_frame, failure)
        )

    # Consumed: from request_idx through response_idx inclusive.
    return response_idx - request_idx + 1


async def _replay_pipeline_block(
    session: Any, scenario: FixtureScenario, block: tuple[int, int]
) -> None:
    """Run a pipelined block: send all >; read N responses; match by tag→id."""
    start, end = block
    block_frames = scenario.frames[start:end]
    request_frames = [f for f in block_frames if f.direction == ">"]
    response_frames = [f for f in block_frames if f.direction == "<"]

    if len(request_frames) != len(response_frames):
        raise FixtureFormatError(
            f"{scenario.path}: pipeline block at lines "
            f"{request_frames[0].line_no}-{response_frames[-1].line_no} "
            f"has {len(request_frames)} requests but {len(response_frames)} responses"
        )

    # Send all requests, capture each request id by tag
    tag_to_request_id: dict[str, int] = {}
    for req in request_frames:
        assert req.method is not None
        assert req.tag is not None  # validated by loader
        rid = await session.send_request_no_wait(req.method, req.payload if req.payload else None)
        tag_to_request_id[req.tag] = rid

    # Read N responses; build id-keyed dict
    id_to_response: dict[int, dict[str, Any]] = {}
    for _ in request_frames:
        resp = await session.read_response()
        rid = resp.get("id")
        if rid is None:
            pytest.fail(f"pipelined response missing 'id' field; got: {_truncate_json(resp)}")
        if rid in id_to_response:
            pytest.fail(f"duplicate response id {rid} in pipelined block")
        id_to_response[rid] = resp

    # Match each expected < against actual by tag → request id
    for resp_frame in response_frames:
        assert resp_frame.tag is not None
        if resp_frame.tag not in tag_to_request_id:
            raise FixtureFormatError(
                f"{scenario.path}:{resp_frame.line_no}: response tag '{resp_frame.tag}' "
                "has no matching request tag"
            )
        expected_id = tag_to_request_id[resp_frame.tag]
        if expected_id not in id_to_response:
            pytest.fail(
                f"pipelined response with id={expected_id} (tag '{resp_frame.tag}') "
                f"not received; actual ids: {sorted(id_to_response.keys())}"
            )
        actual = id_to_response[expected_id]
        failure = match_value(resp_frame.payload, actual)
        if failure is not None:
            pytest.fail(
                f"pipelined replay mismatch (tag '{resp_frame.tag}', id={expected_id}) "
                f"({scenario.path.name}:{resp_frame.line_no})"
                + _format_failure(scenario.path, resp_frame, failure)
            )


# === Tests ===
# Each scenario gets its own test function so failures name the broken
# scenario clearly. DB state setup is inline (per locked §7); fixtures
# stay focused on protocol contract.


@pytest.mark.asyncio
async def test_replay_happy_path(mcp_subprocess_factory, fixture_indexed_db: Path) -> None:
    """S2: positive path that real users walk every session."""
    session = await mcp_subprocess_factory(db_path=fixture_indexed_db)
    try:
        scenario = load_scenario(SESSIONS_DIR / "happy_path.jsonl")
        await replay(session, scenario)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_replay_error_then_recovery(mcp_subprocess_factory, fixture_indexed_db: Path) -> None:
    """S3: a failed validation doesn't leave the session wedged."""
    session = await mcp_subprocess_factory(db_path=fixture_indexed_db)
    try:
        scenario = load_scenario(SESSIONS_DIR / "error_then_recovery.jsonl")
        await replay(session, scenario)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_replay_multiple_back_to_back(
    mcp_subprocess_factory, fixture_indexed_db: Path
) -> None:
    """S4: sequential calls preserve state; no per-call leak."""
    session = await mcp_subprocess_factory(db_path=fixture_indexed_db)
    try:
        scenario = load_scenario(SESSIONS_DIR / "multiple_back_to_back.jsonl")
        await replay(session, scenario)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_replay_pipelined_requests(mcp_subprocess_factory, fixture_indexed_db: Path) -> None:
    """S6: id-correctness under pipelined sends.

    Per the 3.12 pre-implementation spike: the mcp SDK currently
    serializes request processing (one in, one out, in send order).
    The id-mismatch failure mode this scenario is designed to catch
    cannot occur under current SDK behavior — so this test passes
    trivially TODAY. Its role is REGRESSION DEFENSE: if a future SDK
    release introduces concurrent dispatch, the test would catch any
    id-pairing bug. The fixture's header documents this explicitly.
    """
    session = await mcp_subprocess_factory(db_path=fixture_indexed_db)
    try:
        scenario = load_scenario(SESSIONS_DIR / "pipelined_requests.jsonl")
        await replay(session, scenario)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_replay_long_idle_then_call(mcp_subprocess_factory, fixture_indexed_db: Path) -> None:
    """S1: (a) SDK stdio handler survives idle.

    Failure mode: subsequent tools/call returns timeout, malformed
    response, or server has exited. If failure occurs at exactly 30s
    with connection error, likely SDK internal idle timeout —
    investigate.
    """
    session = await mcp_subprocess_factory(db_path=fixture_indexed_db)
    try:
        scenario = load_scenario(SESSIONS_DIR / "long_idle_then_call.jsonl")
        await replay(session, scenario)
    finally:
        await session.close()


# === Loader self-tests ===
# A small handful to catch loader regressions; not full coverage of the
# fixture format (that's exercised by the scenario tests).


def test_loader_rejects_unclosed_pipeline_block(tmp_path: Path) -> None:
    """Format error: #PIPELINE-START without END."""
    fp = tmp_path / "bad.jsonl"
    fp.write_text(
        '#PIPELINE-START\n> tools/call {"name":"recent","arguments":{"limit":3}}  # tag:a\n',
        encoding="utf-8",
    )
    with pytest.raises(FixtureFormatError, match="unclosed #PIPELINE-START"):
        load_scenario(fp)


def test_loader_rejects_duplicate_tags_in_pipeline_block(tmp_path: Path) -> None:
    """Format error: duplicate tag within a pipeline block."""
    fp = tmp_path / "bad.jsonl"
    fp.write_text(
        "#PIPELINE-START\n"
        '> tools/call {"name":"recent","arguments":{"limit":3}}  # tag:dup\n'
        '> tools/call {"name":"command_stats","arguments":{"pattern":"git"}}  # tag:dup\n'
        '< {"jsonrpc":"2.0","id":1,"result":{}}  # tag:dup\n'
        '< {"jsonrpc":"2.0","id":2,"result":{}}  # tag:dup\n'
        "#PIPELINE-END\n",
        encoding="utf-8",
    )
    with pytest.raises(FixtureFormatError, match="duplicate tag 'dup'"):
        load_scenario(fp)


def test_loader_rejects_pipelined_request_without_tag(tmp_path: Path) -> None:
    """Format error: pipelined > requires '# tag:<name>' annotation."""
    fp = tmp_path / "bad.jsonl"
    fp.write_text(
        "#PIPELINE-START\n"
        '> tools/call {"name":"recent","arguments":{"limit":3}}\n'
        '< {"jsonrpc":"2.0","id":1,"result":{}}  # tag:foo\n'
        "#PIPELINE-END\n",
        encoding="utf-8",
    )
    with pytest.raises(FixtureFormatError, match="requires '# tag"):
        load_scenario(fp)


def test_loader_rejects_unbalanced_tags(tmp_path: Path) -> None:
    """Format error: tag on > with no matching < (or vice versa)."""
    fp = tmp_path / "bad.jsonl"
    fp.write_text(
        "#PIPELINE-START\n"
        '> tools/call {"name":"recent","arguments":{"limit":3}}  # tag:a\n'
        '> tools/call {"name":"command_stats","arguments":{"pattern":"git"}}  # tag:b\n'
        '< {"jsonrpc":"2.0","id":1,"result":{}}  # tag:a\n'
        '< {"jsonrpc":"2.0","id":2,"result":{}}  # tag:c\n'
        "#PIPELINE-END\n",
        encoding="utf-8",
    )
    with pytest.raises(FixtureFormatError, match="tag mismatch"):
        load_scenario(fp)


def test_loader_parses_idle_directive(tmp_path: Path) -> None:
    """#IDLE <seconds> attached to the next frame index."""
    fp = tmp_path / "ok.jsonl"
    fp.write_text(
        '> initialize {"protocolVersion":"x"}\n'
        '< {"jsonrpc":"2.0","id":1,"result":{}}\n'
        "#IDLE 5\n"
        '> tools/call {"name":"recent","arguments":{"limit":1}}\n'
        '< {"jsonrpc":"2.0","id":2,"result":{}}\n',
        encoding="utf-8",
    )
    sc = load_scenario(fp)
    # The third frame (index 2) is the request after #IDLE 5
    assert sc.idle_at == {2: 5}


def test_matcher_sentinel_TS() -> None:
    assert match_value(_SENTINEL_TS, 100) is None
    assert match_value(_SENTINEL_TS, 0) is None
    assert match_value(_SENTINEL_TS, -1) is not None
    assert match_value(_SENTINEL_TS, "100") is not None


def test_matcher_sentinel_HEX16() -> None:
    assert match_value(_SENTINEL_HEX16, "0123456789abcdef") is None
    assert match_value(_SENTINEL_HEX16, "0123") is not None  # too short
    assert match_value(_SENTINEL_HEX16, "0123456789ABCDEF") is not None  # uppercase


def test_matcher_sentinel_SCORE() -> None:
    assert match_value(_SENTINEL_SCORE, 0.5) is None
    assert match_value(_SENTINEL_SCORE, 1.0) is None
    assert match_value(_SENTINEL_SCORE, -1.0) is None
    assert match_value(_SENTINEL_SCORE, 1.5) is not None


def test_matcher_sentinel_INT() -> None:
    assert match_value(_SENTINEL_INT, 5) is None
    assert match_value(_SENTINEL_INT, -5) is None
    assert match_value(_SENTINEL_INT, True) is not None  # booleans rejected
    assert match_value(_SENTINEL_INT, "5") is not None


def test_matcher_sentinel_ANY() -> None:
    assert match_value(_SENTINEL_ANY, 42) is None
    assert match_value(_SENTINEL_ANY, "anything") is None
    assert match_value(_SENTINEL_ANY, {"x": 1}) is None


def test_matcher_path_reported_on_mismatch() -> None:
    """Mismatch failure includes JSON path."""
    failure = match_value({"a": {"b": 1}}, {"a": {"b": 2}})
    assert failure is not None
    assert failure.path == "$.a.b"
