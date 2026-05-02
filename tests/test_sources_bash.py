"""Tests for ``recall.sources.bash``."""

from __future__ import annotations

from pathlib import Path

from recall.sources.bash import BashSource


def test_bash_plain_no_timestamps(tmp_path: Path) -> None:
    f = tmp_path / "bash_history"
    f.write_text("ls -la\ncd /tmp\necho hi\n", encoding="utf-8")
    entries = list(BashSource(path=f).iter_entries())
    assert [e.text for e in entries] == ["ls -la", "cd /tmp", "echo hi"]
    assert all(e.ts == 0 for e in entries)
    assert all(e.source == "bash" for e in entries)


def test_bash_with_histtimeformat_prefix(tmp_path: Path) -> None:
    f = tmp_path / "bash_history"
    f.write_text(
        "#1700000000\nls -la\n#1700000010\ncd /tmp\n#1700000020\necho hi\n",
        encoding="utf-8",
    )
    entries = list(BashSource(path=f).iter_entries())
    assert [e.text for e in entries] == ["ls -la", "cd /tmp", "echo hi"]
    assert entries[0].ts == 1700000000
    assert entries[1].ts == 1700000010
    assert entries[2].ts == 1700000020


def test_bash_short_digit_line_is_not_a_timestamp(tmp_path: Path) -> None:
    """A line like '#5' could be a comment, not a HISTTIMEFORMAT marker.
    We require >= 9 digits (timestamps from ~2001 onward) to disambiguate."""
    f = tmp_path / "bash_history"
    f.write_text("#5\nreal command\n", encoding="utf-8")
    entries = list(BashSource(path=f).iter_entries())
    # '#5' treated as a real command, then 'real command' as the next.
    assert [e.text for e in entries] == ["#5", "real command"]
    assert entries[0].ts == 0


def test_bash_since_filter_known_ts(tmp_path: Path) -> None:
    f = tmp_path / "bash_history"
    f.write_text(
        "#1700000000\nold\n#1700000010\nnewer\n#1700000020\nnewest\n",
        encoding="utf-8",
    )
    entries = list(BashSource(path=f).iter_entries(since=1700000000))
    assert [e.text for e in entries] == ["newer", "newest"]


def test_bash_unknown_ts_always_yields(tmp_path: Path) -> None:
    f = tmp_path / "bash_history"
    f.write_text("plain1\n#1700000000\ntimestamped\nplain2\n", encoding="utf-8")
    entries = list(BashSource(path=f).iter_entries(since=2_000_000_000))
    # plain1 (ts=0) and plain2 (ts=0 — pending was consumed by 'timestamped')
    # yield; 'timestamped' (ts=1700000000) is filtered out.
    assert [e.text for e in entries] == ["plain1", "plain2"]


def test_bash_latin1_fallback(tmp_path: Path) -> None:
    f = tmp_path / "bash_history"
    f.write_bytes(b"echo \xff\nls -la\n")
    entries = list(BashSource(path=f).iter_entries())
    assert len(entries) == 2
    assert entries[1].text == "ls -la"


def test_bash_missing_file_returns_empty(tmp_path: Path) -> None:
    src = BashSource(path=tmp_path / "does-not-exist")
    assert list(src.iter_entries()) == []
