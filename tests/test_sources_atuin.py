"""Tests for ``recall.sources.atuin``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from recall.sources.atuin import AtuinSchemaError, AtuinSource


def test_atuin_reads_all_entries(atuin_fixture_path: Path) -> None:
    entries = list(AtuinSource(path=atuin_fixture_path).iter_entries())
    assert len(entries) == 5
    first = entries[0]
    assert first.text == "ls -la"
    assert first.ts == 1700000000  # ns → s conversion
    assert first.cwd == "/home/user"
    assert first.hostname == "hostX"
    assert first.exit_code == 0
    assert first.duration_ms == 1500  # 1500_000000 ns → 1500 ms
    assert first.session_id == "sess-A"
    assert first.source_id == "id-001"
    assert first.source == "atuin"


def test_atuin_partial_metadata_row(atuin_fixture_path: Path) -> None:
    """The fifth row has duration=NULL exit=NULL — must surface as None."""
    entries = list(AtuinSource(path=atuin_fixture_path).iter_entries())
    last = entries[4]
    assert last.text == "echo hello"
    assert last.duration_ms is None
    assert last.exit_code is None
    assert last.cwd == "/home/user"


def test_atuin_since_filter(atuin_fixture_path: Path) -> None:
    entries = list(AtuinSource(path=atuin_fixture_path).iter_entries(since=1700000010))
    # ts > 1700000010 → keeps id-003, id-004, id-005
    assert [e.source_id for e in entries] == ["id-003", "id-004", "id-005"]


def test_atuin_missing_required_column_raises(tmp_path: Path) -> None:
    db = tmp_path / "broken.db"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE history (id TEXT, command TEXT)")
    conn.commit()
    conn.close()

    src = AtuinSource(path=db)
    with pytest.raises(AtuinSchemaError, match="timestamp"):
        list(src.iter_entries())


def test_atuin_minimal_required_schema_works(tmp_path: Path) -> None:
    """Only the required columns (timestamp, command) — optional ones absent.
    Must still parse, with optional Entry fields = None."""
    db = tmp_path / "minimal.db"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE history (timestamp INTEGER, command TEXT)")
    conn.execute(
        "INSERT INTO history (timestamp, command) VALUES (?, ?)",
        (1700000000_000000000, "ls"),
    )
    conn.commit()
    conn.close()

    entries = list(AtuinSource(path=db).iter_entries())
    assert len(entries) == 1
    e = entries[0]
    assert e.text == "ls"
    assert e.ts == 1700000000
    assert e.cwd is None
    assert e.hostname is None
    assert e.duration_ms is None
    assert e.exit_code is None
    assert e.session_id is None
    assert e.source_id is None


def test_atuin_readonly_immutable_no_journal_sidecars(
    atuin_fixture_path: Path,
) -> None:
    """The CLAUDE.md §3 rule: opening with ``mode=ro&immutable=1`` MUST
    NOT create -wal / -shm / -journal sidecar files next to the user's
    atuin DB. Iterating to exhaustion is what triggers any sidecar
    creation we'd accidentally do."""
    src = AtuinSource(path=atuin_fixture_path)
    list(src.iter_entries())  # exhaust the cursor

    parent = atuin_fixture_path.parent
    sidecars = sorted(p.name for p in parent.iterdir() if p.name != atuin_fixture_path.name)
    assert sidecars == [], f"unexpected sidecars created: {sidecars}"


def test_atuin_missing_file_returns_empty(tmp_path: Path) -> None:
    src = AtuinSource(path=tmp_path / "does-not-exist")
    assert list(src.iter_entries()) == []


def test_atuin_order_by_timestamp(atuin_fixture_path: Path) -> None:
    entries = list(AtuinSource(path=atuin_fixture_path).iter_entries())
    timestamps = [e.ts for e in entries]
    assert timestamps == sorted(timestamps)
