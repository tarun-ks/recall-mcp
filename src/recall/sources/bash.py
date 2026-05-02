"""Reader for bash history files (default: ``~/.bash_history``).

Handles plain commands and ``HISTTIMEFORMAT``-prefixed timestamp lines
(``#<unix_seconds>`` immediately preceding the command). Multi-line bash
commands are emitted as separate entries — fidelity loss is accepted as a
known limitation; multi-line is rare in practice. Latin-1 fallback on
invalid UTF-8.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from pathlib import Path

from recall.sources.base import Entry

_LOG = logging.getLogger("recall.sources.bash")

_DEFAULT_PATH = Path.home() / ".bash_history"

# HISTTIMEFORMAT timestamp marker. Require ≥ 9 digits (timestamps from
# ~2001 onward) so a line like '#5' isn't mistaken for a ts.
_TS_LINE_RE = re.compile(r"^#(\d{9,})$")


class BashSource:
    """``HistorySource`` over a bash history file."""

    name = "bash"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path if path is not None else _DEFAULT_PATH

    def iter_entries(self, since: int | None = None) -> Iterator[Entry]:
        if not self.path.exists():
            return
        threshold = since if since is not None else 0
        text = _read_with_fallback(self.path)
        pending_ts = 0
        for line in text.split("\n"):
            if not line:
                continue
            m = _TS_LINE_RE.match(line)
            if m:
                pending_ts = int(m.group(1))
                continue
            ts = pending_ts
            pending_ts = 0  # consumed
            if ts > 0 and ts <= threshold:
                continue
            yield Entry(text=line, ts=ts, source=self.name)


def _read_with_fallback(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        _LOG.warning(
            "recall.sources.bash: %s contains invalid UTF-8; falling back to latin-1",
            path,
        )
        return path.read_text(encoding="latin-1")


__all__ = ("BashSource",)
