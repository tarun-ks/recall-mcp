"""Reader for zsh history files (default: ``~/.zsh_history``).

Handles both ``EXTENDED_HISTORY`` (``: <ts>:<dur>;<command>``) and plain
formats, multi-line commands joined by trailing-backslash continuation,
and invalid UTF-8 (latin-1 fallback). Logs and skips malformed lines —
never crashes.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from recall.sources.base import Entry

_LOG = logging.getLogger("recall.sources.zsh")

_DEFAULT_PATH = Path.home() / ".zsh_history"

# Extended-history line shape: ': <ts>:<dur>;<command>'
# `<ts>` and `<dur>` are unsigned integer seconds.
_EXTENDED_RE = re.compile(r"^:\s+(\d+):(\d+);(.*)$", re.DOTALL)
# A line that LOOKS like an attempt at extended format (starts with ': ')
# but doesn't parse — log + skip rather than mistaking it for plain.
_LOOKS_EXTENDED_RE = re.compile(r"^:\s")


class ZshSource:
    """``HistorySource`` over a zsh history file."""

    name = "zsh"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else _DEFAULT_PATH

    def iter_entries(self, since: int | None = None) -> Iterator[Entry]:
        if not self.path.exists():
            return
        threshold = since if since is not None else 0
        text = _read_with_fallback(self.path)
        for raw in _join_continuations(text):
            entry = self._parse(raw)
            if entry is None:
                continue
            # ts == 0 always passes (unknown); known ts <= threshold is filtered.
            if entry.ts > 0 and entry.ts <= threshold:
                continue
            yield entry

    def _parse(self, line: str) -> Entry | None:
        m = _EXTENDED_RE.match(line)
        if m:
            ts = int(m.group(1))
            duration_s = int(m.group(2))
            return Entry(
                text=m.group(3),
                ts=ts,
                source=self.name,
                duration_ms=duration_s * 1000,
            )
        if _LOOKS_EXTENDED_RE.match(line):
            _LOG.warning(
                "recall.sources.zsh: looks-extended-but-malformed, skipping: %r",
                line[:100],
            )
            return None
        return Entry(text=line, ts=0, source=self.name)


def _read_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        _LOG.warning(
            "recall.sources.zsh: %s contains invalid UTF-8; falling back to latin-1",
            path,
        )
        return path.read_text(encoding="latin-1")


def _join_continuations(text: str) -> Iterator[str]:
    """Yield logical lines, joining trailing-backslash continuations.

    zsh stores multi-line commands with embedded ``\\<newline>`` between
    physical lines. When reading, a trailing ``\\`` means the next line
    continues this one; we drop the backslash and rejoin with a real
    newline so the parsed Entry.text reflects the user's input.
    """
    physical = text.split("\n")
    i = 0
    while i < len(physical):
        line = physical[i]
        while line.endswith("\\") and i + 1 < len(physical):
            line = line[:-1] + "\n" + physical[i + 1]
            i += 1
        i += 1
        if line:
            yield line


__all__ = ("ZshSource",)
