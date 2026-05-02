"""Tests for ``recall.sources.zsh``."""

from __future__ import annotations

import logging
from pathlib import Path

from recall.sources.zsh import ZshSource


def test_zsh_extended_history(tmp_path: Path) -> None:
    f = tmp_path / "zsh_history"
    f.write_text(": 1700000000:5;ls -la\n: 1700000010:1;cd /tmp\n", encoding="utf-8")
    entries = list(ZshSource(path=f).iter_entries())
    assert len(entries) == 2
    assert entries[0].text == "ls -la"
    assert entries[0].ts == 1700000000
    assert entries[0].duration_ms == 5000
    assert entries[0].source == "zsh"
    assert entries[1].text == "cd /tmp"
    assert entries[1].ts == 1700000010
    assert entries[1].duration_ms == 1000


def test_zsh_plain_format_emits_ts_zero(tmp_path: Path) -> None:
    f = tmp_path / "zsh_history"
    f.write_text("ls -la\ncd /tmp\n", encoding="utf-8")
    entries = list(ZshSource(path=f).iter_entries())
    assert len(entries) == 2
    assert entries[0].text == "ls -la"
    assert entries[0].ts == 0
    assert entries[0].duration_ms is None


def test_zsh_multiline_extended(tmp_path: Path) -> None:
    """A multi-line zsh entry uses trailing-backslash continuation; the
    physical lines must rejoin to a single Entry whose text contains
    embedded newlines."""
    f = tmp_path / "zsh_history"
    f.write_text(
        ": 1700000000:5;cat <<EOF\\\nhello\\\nworld\\\nEOF\n: 1700000010:0;ls\n",
        encoding="utf-8",
    )
    entries = list(ZshSource(path=f).iter_entries())
    assert len(entries) == 2
    assert "hello" in entries[0].text
    assert "world" in entries[0].text
    assert "EOF" in entries[0].text
    assert entries[1].text == "ls"


def test_zsh_latin1_fallback(tmp_path: Path) -> None:
    """A byte invalid in UTF-8 (0xFF) must not crash; latin-1 fallback yields entries."""
    f = tmp_path / "zsh_history"
    f.write_bytes(b": 1700000000:0;echo \xff\n: 1700000010:0;ls\n")
    entries = list(ZshSource(path=f).iter_entries())
    assert len(entries) == 2
    assert entries[1].text == "ls"


def test_zsh_malformed_extended_logs_and_skips(tmp_path: Path, caplog: object) -> None:
    """A line that looks-extended but doesn't parse must be logged + skipped,
    not silently emitted as plain (it'd be garbage)."""
    f = tmp_path / "zsh_history"
    f.write_text(
        ": notvalid:badbad;malformed line\n: 1700000000:5;valid line\n",
        encoding="utf-8",
    )
    caplog_inst = caplog  # for typing; pytest provides LogCaptureFixture
    with caplog_inst.at_level(logging.WARNING):  # type: ignore[attr-defined]
        entries = list(ZshSource(path=f).iter_entries())
    assert len(entries) == 1
    assert entries[0].text == "valid line"
    assert any(
        "looks-extended-but-malformed" in r.message
        for r in caplog_inst.records  # type: ignore[attr-defined]
    )


def test_zsh_since_filter_excludes_old_known_ts(tmp_path: Path) -> None:
    f = tmp_path / "zsh_history"
    f.write_text(
        ": 1700000000:0;old\n: 1700000010:0;newer\n: 1700000020:0;newest\n",
        encoding="utf-8",
    )
    entries = list(ZshSource(path=f).iter_entries(since=1700000000))
    assert [e.text for e in entries] == ["newer", "newest"]


def test_zsh_unknown_ts_always_yields_regardless_of_since(tmp_path: Path) -> None:
    """Plain entries (ts=0) bypass the since filter — we can't tell if
    they're old or new, so the indexer's UNIQUE constraint disambiguates."""
    f = tmp_path / "zsh_history"
    f.write_text("plain command\n: 1700000000:0;extended\n", encoding="utf-8")
    entries = list(ZshSource(path=f).iter_entries(since=2_000_000_000))
    assert len(entries) == 1
    assert entries[0].text == "plain command"
    assert entries[0].ts == 0


def test_zsh_missing_file_returns_empty(tmp_path: Path) -> None:
    src = ZshSource(path=tmp_path / "does-not-exist")
    assert list(src.iter_entries()) == []


def test_zsh_empty_lines_skipped(tmp_path: Path) -> None:
    f = tmp_path / "zsh_history"
    f.write_text("\n\n: 1700000000:0;real\n\n", encoding="utf-8")
    entries = list(ZshSource(path=f).iter_entries())
    assert [e.text for e in entries] == ["real"]
