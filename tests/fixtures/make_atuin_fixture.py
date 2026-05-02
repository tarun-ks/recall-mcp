"""Build a deterministic atuin-shaped sqlite DB for tests.

Used by ``tests/conftest.py`` (a session-scoped pytest fixture rebuilds it
into ``tmp_path``). Also runnable as a script for human inspection:

    uv run python tests/fixtures/make_atuin_fixture.py

This writes ``tests/fixtures/atuin_sample.db`` next to the script. The
file is gitignored — we don't commit binary fixtures, we regenerate.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Five rows with a mix of full / partial metadata so tests can assert
# the optional-column handling. Timestamps are in nanoseconds (atuin's unit).
# Tuple shape: (id, timestamp_ns, duration_ns, exit, command, cwd, session, hostname)
_Row = tuple[str, int, int | None, int | None, str, str | None, str | None, str | None]

_ROWS: tuple[_Row, ...] = (
    ("id-001", 1700000000_000000000, 1500_000000, 0, "ls -la", "/home/user", "sess-A", "hostX"),
    ("id-002", 1700000010_000000000, 200_000000, 0, "cd /tmp", "/home/user", "sess-A", "hostX"),
    (
        "id-003",
        1700000020_000000000,
        5000_000000,
        1,
        "python failing.py",
        "/tmp",
        "sess-A",
        "hostX",
    ),
    (
        "id-004",
        1700000030_000000000,
        800_000000,
        0,
        "git status",
        "/home/user/repo",
        "sess-B",
        "hostY",
    ),
    # Partial-metadata row: duration / exit unknown, cwd present.
    ("id-005", 1700000040_000000000, None, None, "echo hello", "/home/user", "sess-B", "hostY"),
)


def make_atuin_fixture(path: Path) -> None:
    """(Re)create an atuin-shaped sqlite DB at ``path`` with deterministic content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE history (
                id        TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                duration  INTEGER,
                exit      INTEGER,
                command   TEXT NOT NULL,
                cwd       TEXT,
                session   TEXT,
                hostname  TEXT
            );
            """
        )
        conn.executemany(
            "INSERT INTO history "
            "(id, timestamp, duration, exit, command, cwd, session, hostname) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            _ROWS,
        )
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    target = Path(__file__).parent / "atuin_sample.db"
    make_atuin_fixture(target)
    print(f"wrote {target}")
