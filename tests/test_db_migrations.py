"""Tests for ``recall.db``: connect, migrate, meta, salt, dedup hash, FTS, vec."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import pytest

from recall.db import (
    DBError,
    connect,
    dedup_hash,
    dedup_salt,
    get_meta,
    migrate,
    rotate_dedup_salt,
    schema_version,
    set_meta,
    transaction,
)


@pytest.fixture()
def db(tmp_path: Path) -> sqlite3.Connection:
    """Open a fresh DB in a tmp path and run migrate. Closes on teardown."""
    conn = connect(tmp_path / "recall.sqlite")
    migrate(conn)
    return conn


# === Connect / migrate / version ===


def test_connect_uses_env_override_when_no_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RECALL_DB_PATH overrides the built-in default when no path arg given."""
    target = tmp_path / "from_env.sqlite"
    monkeypatch.setenv("RECALL_DB_PATH", str(target))
    conn = connect()
    migrate(conn)
    assert target.exists()


def test_connect_explicit_arg_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit path arg beats RECALL_DB_PATH (resolution order: arg > env > default)."""
    monkeypatch.setenv("RECALL_DB_PATH", str(tmp_path / "from_env.sqlite"))
    explicit = tmp_path / "from_arg.sqlite"
    conn = connect(explicit)
    migrate(conn)
    assert explicit.exists()
    assert not (tmp_path / "from_env.sqlite").exists()


def test_connect_env_path_expands_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``~`` in RECALL_DB_PATH must be expanded relative to $HOME."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("RECALL_DB_PATH", "~/custom/recall.sqlite")
    conn = connect()
    migrate(conn)
    assert (tmp_path / "custom" / "recall.sqlite").exists()


def test_connect_loads_sqlite_vec(tmp_path: Path) -> None:
    conn = connect(tmp_path / "vec.sqlite")
    row = conn.execute("SELECT vec_version()").fetchone()
    assert row[0].startswith("v")


def test_connect_sets_wal_journal_mode(tmp_path: Path) -> None:
    conn = connect(tmp_path / "wal.sqlite")
    row = conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].lower() == "wal"


def test_connect_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c" / "recall.sqlite"
    assert not nested.parent.exists()
    connect(nested)
    assert nested.parent.is_dir()
    assert nested.exists()


def test_fresh_init_creates_schema_v1(tmp_path: Path) -> None:
    conn = connect(tmp_path / "fresh.sqlite")
    assert schema_version(conn) == 0  # before migrate
    migrate(conn)
    assert schema_version(conn) == 1


def test_migrate_idempotent(db: sqlite3.Connection) -> None:
    salt_before = get_meta(db, "dedup_salt")
    migrate(db)
    migrate(db)
    assert schema_version(db) == 1
    assert get_meta(db, "dedup_salt") == salt_before


def test_migrate_rejects_future_schema(tmp_path: Path) -> None:
    conn = connect(tmp_path / "future.sqlite")
    conn.execute("PRAGMA user_version = 999")
    with pytest.raises(DBError, match="schema is at v999"):
        migrate(conn)


# === Meta accessors ===


def test_meta_seeded_after_migrate(db: sqlite3.Connection) -> None:
    assert get_meta(db, "dedup_salt") is not None
    assert get_meta(db, "dedup_salt_version") == "1"
    assert get_meta(db, "scrubber_version") is not None
    assert get_meta(db, "schema_version") == "1"


def test_get_meta_missing_returns_none(db: sqlite3.Connection) -> None:
    assert get_meta(db, "no-such-key") is None


def test_get_meta_returns_none_when_table_missing(tmp_path: Path) -> None:
    """Defensive: get_meta on a fresh DB (no migrate yet) shouldn't crash."""
    conn = connect(tmp_path / "noschema.sqlite")
    assert get_meta(conn, "anything") is None


def test_set_meta_idempotent_overwrites(db: sqlite3.Connection) -> None:
    set_meta(db, "k", "first")
    assert get_meta(db, "k") == "first"
    set_meta(db, "k", "second")
    assert get_meta(db, "k") == "second"


# === Dedup salt ===


def test_dedup_salt_returns_32_bytes(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    assert isinstance(salt, bytes)
    assert len(salt) == 32


def test_dedup_salt_stable_across_calls(db: sqlite3.Connection) -> None:
    assert dedup_salt(db) == dedup_salt(db)


def test_dedup_salt_preserved_across_reconnects(tmp_path: Path) -> None:
    """Salt must persist across process restarts — same hashes, same DB file."""
    path = tmp_path / "preserve.sqlite"
    c1 = connect(path)
    migrate(c1)
    salt1 = dedup_salt(c1)
    c1.close()

    c2 = connect(path)
    migrate(c2)
    salt2 = dedup_salt(c2)
    assert salt1 == salt2


def test_dedup_salt_missing_meta_raises(tmp_path: Path) -> None:
    conn = connect(tmp_path / "nomigrate.sqlite")
    # Schema not migrated; meta table doesn't exist yet.
    with pytest.raises(DBError, match="missing from meta"):
        dedup_salt(conn)


def test_rotate_dedup_salt_changes_value_and_bumps_version(db: sqlite3.Connection) -> None:
    old = dedup_salt(db)
    old_v = int(get_meta(db, "dedup_salt_version") or "0")
    new = rotate_dedup_salt(db)
    assert new != old
    assert dedup_salt(db) == new
    new_v = int(get_meta(db, "dedup_salt_version") or "0")
    assert new_v == old_v + 1


def test_rotate_dedup_salt_does_not_clear_commands(db: sqlite3.Connection) -> None:
    """Rotate is a primitive — clearing rows is the CLI's job. Verify
    that rotate alone leaves rows in place, since this property is what
    makes the CLI's --new-salt-without-rebuild check load-bearing."""
    salt = dedup_salt(db)
    db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "ls", dedup_hash(salt, "ls"), 1700000000),
    )
    assert db.execute("SELECT count(*) FROM commands").fetchone()[0] == 1
    rotate_dedup_salt(db)
    assert db.execute("SELECT count(*) FROM commands").fetchone()[0] == 1


# === Dedup hash ===


def test_dedup_hash_size_32(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    h = dedup_hash(salt, "ls -la")
    assert len(h) == 32
    assert isinstance(h, bytes)


def test_dedup_hash_deterministic() -> None:
    salt = b"\x00" * 32
    assert dedup_hash(salt, "ls -la") == dedup_hash(salt, "ls -la")


def test_dedup_hash_input_dependent() -> None:
    salt = b"\x00" * 32
    assert dedup_hash(salt, "a") != dedup_hash(salt, "b")


def test_dedup_hash_salt_dependent() -> None:
    """Different salts → different hashes for the same raw text. This is
    what makes the salt a namespace separator across Recall installs."""
    a = dedup_hash(b"\x00" * 32, "ls -la")
    b = dedup_hash(b"\x01" * 32, "ls -la")
    assert a != b


def test_dedup_hash_unicode() -> None:
    salt = b"\x00" * 32
    h = dedup_hash(salt, "echo héllo ✨")
    assert len(h) == 32


# === Commands table + uniqueness ===


def test_commands_unique_on_source_hash_ts(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    h = dedup_hash(salt, "ls")
    db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "ls", h, 1700000000),
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
            ("zsh", "ls", h, 1700000000),
        )


def test_commands_unique_distinguishes_source(db: sqlite3.Connection) -> None:
    """Same hash + ts but different source: allowed (atuin and zsh see the
    same command)."""
    salt = dedup_salt(db)
    h = dedup_hash(salt, "ls")
    db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "ls", h, 1700000000),
    )
    db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("atuin", "ls", h, 1700000000),
    )
    assert db.execute("SELECT count(*) FROM commands").fetchone()[0] == 2


# === FTS5 sync triggers ===


def test_fts_insert_trigger(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "git checkout main", dedup_hash(salt, "git checkout main"), 1700000000),
    )
    rows = db.execute(
        "SELECT rowid FROM commands_fts WHERE commands_fts MATCH 'checkout'"
    ).fetchall()
    assert len(rows) == 1


def test_fts_delete_trigger(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    cur = db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "deletable thing", dedup_hash(salt, "deletable thing"), 1700000001),
    )
    rowid = cur.lastrowid
    assert (
        db.execute(
            "SELECT count(*) FROM commands_fts WHERE commands_fts MATCH 'deletable'"
        ).fetchone()[0]
        == 1
    )
    db.execute("DELETE FROM commands WHERE id = ?", (rowid,))
    assert (
        db.execute(
            "SELECT count(*) FROM commands_fts WHERE commands_fts MATCH 'deletable'"
        ).fetchone()[0]
        == 0
    )


def test_fts_update_trigger(db: sqlite3.Connection) -> None:
    salt = dedup_salt(db)
    cur = db.execute(
        "INSERT INTO commands (source, text_scrubbed, text_hash, ts) VALUES (?, ?, ?, ?)",
        ("zsh", "old text", dedup_hash(salt, "old text"), 1700000002),
    )
    rowid = cur.lastrowid
    db.execute("UPDATE commands SET text_scrubbed = ? WHERE id = ?", ("brand new", rowid))
    assert (
        db.execute("SELECT count(*) FROM commands_fts WHERE commands_fts MATCH 'old'").fetchone()[0]
        == 0
    )
    assert (
        db.execute("SELECT count(*) FROM commands_fts WHERE commands_fts MATCH 'brand'").fetchone()[
            0
        ]
        == 1
    )


# === sqlite-vec virtual table ===


def test_vec_table_accepts_384d_embedding(db: sqlite3.Connection) -> None:
    """Insert a real 384-float embedding; this is the dimension we ship."""
    embedding = struct.pack("=384f", *([0.01] * 384))
    db.execute("INSERT INTO commands_vec(command_id, embedding) VALUES (1, ?)", (embedding,))
    assert db.execute("SELECT count(*) FROM commands_vec").fetchone()[0] == 1


def test_vec_table_rejects_wrong_dim(db: sqlite3.Connection) -> None:
    """vec0 is dimension-typed; wrong-dim insert must fail. This is what
    keeps embedding-version-mismatch from silently producing garbage."""
    bad = struct.pack("=128f", *([0.01] * 128))
    with pytest.raises(sqlite3.OperationalError):
        db.execute("INSERT INTO commands_vec(command_id, embedding) VALUES (2, ?)", (bad,))


# === transaction context manager ===


def test_transaction_commits_on_success(db: sqlite3.Connection) -> None:
    with transaction(db):
        set_meta(db, "tx_test", "ok")
    assert get_meta(db, "tx_test") == "ok"


def test_transaction_rolls_back_on_exception(db: sqlite3.Connection) -> None:
    with pytest.raises(RuntimeError, match="boom"), transaction(db):
        set_meta(db, "tx_rollback", "should-not-persist")
        raise RuntimeError("boom")
    assert get_meta(db, "tx_rollback") is None
