"""Integration tests for the indexer (Commit 2.8).

The indexer is consumer-less until Phase 3's MCP server lands — there is
no public ``recall search`` yet. These tests verify the DB-layer write
path directly: rows land in ``commands``, vectors land in ``commands_vec``,
the dedup constraint holds, the rebuild path drops + recreates correctly,
the per-source cursor advances, and (critically per Q8) the scrubber's
write-path integration leaves zero matches against any of the 18
scrubber-pattern types in ``commands.text_scrubbed``.

All tests use a stub Embedder that produces deterministic vectors per
text — no model load, fast lane. End-to-end semantic correctness is
already covered by the eval-lane tests on the SemanticRanker side.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Iterator, Sequence

import numpy as np
import pytest

from recall.db import (
    connect,
    dedup_salt,
    get_cursor,
    get_meta,
    migrate,
    rotate_dedup_salt,
)
from recall.indexer import index_sources, rebuild
from recall.sources.base import Entry


class _FakeEmbedder384:
    """Deterministic in-memory embedder satisfying Embedder's duck-typed
    surface, producing dim=384 vectors to match the on-disk schema's
    ``commands_vec.embedding FLOAT[384]``. SHA-256 of text → 16 bytes →
    repeated to 384 floats, centered + L2-normalized. No model load."""

    model_name = "fake-embedder/v0"
    model_revision = None
    dim = 384

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            small = np.frombuffer(digest[:16], dtype=np.uint8).astype(np.float32)
            small -= 128.0
            tiled = np.tile(small, 24).astype(np.float32)  # 16 * 24 = 384
            n = float(np.linalg.norm(tiled))
            out[i] = tiled / (n if n > 0 else 1.0)
        return out


class _StubSource:
    """A HistorySource that yields pre-defined entries; honors ``since`` so
    cursor tests can exercise incremental behavior."""

    def __init__(self, name: str, entries: list[Entry]) -> None:
        self.name = name
        self._entries = entries

    def iter_entries(self, since: int | None = None) -> Iterator[Entry]:
        for e in self._entries:
            if e.ts == 0 or since is None or e.ts > since:
                yield e


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    """Per-test fresh DB at tmp_path/db.sqlite, migrated to v1."""
    db_path = tmp_path / "db.sqlite"
    c = connect(db_path)
    migrate(c)
    yield c
    c.close()


@pytest.fixture
def real_dim_embedder() -> _FakeEmbedder384:
    return _FakeEmbedder384()


def _row_count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


# === 1. Basic round-trip ===


def test_index_inserts_rows_and_vectors(conn, real_dim_embedder) -> None:
    """Single source, three entries → three rows in commands + commands_vec."""
    src = _StubSource(
        "zsh",
        [
            Entry(text="ls -la", ts=100, source="zsh"),
            Entry(text="git status", ts=200, source="zsh"),
            Entry(text="cd /tmp", ts=300, source="zsh"),
        ],
    )
    result = index_sources(conn, [src], embedder=real_dim_embedder, progress=False)

    assert result.inserted == 3
    assert result.skipped_dedup == 0
    assert _row_count(conn, "commands") == 3
    assert _row_count(conn, "commands_vec") == 3
    # Cursor advanced to max ts in the batch.
    assert get_cursor(conn, "zsh") == 300


# === 2. Incremental: re-indexing yields no new rows ===


def test_reindex_with_no_new_entries_inserts_nothing(conn, real_dim_embedder) -> None:
    src = _StubSource(
        "zsh",
        [
            Entry(text="cmd one", ts=100, source="zsh"),
            Entry(text="cmd two", ts=200, source="zsh"),
        ],
    )
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 2

    # Re-run: cursor now at 200, source's iter_entries(since=200) yields nothing.
    result2 = index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert result2.inserted == 0
    assert _row_count(conn, "commands") == 2


# === 3. Rebuild clears + refills, salt preserved ===


def test_rebuild_clears_and_refills_preserving_salt(conn, real_dim_embedder) -> None:
    src = _StubSource(
        "zsh",
        [Entry(text=f"cmd {i}", ts=100 + i, source="zsh") for i in range(5)],
    )
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 5

    salt_before = dedup_salt(conn)
    salt_version_before = get_meta(conn, "dedup_salt_version")

    rebuild(conn)
    assert _row_count(conn, "commands") == 0
    assert _row_count(conn, "commands_vec") == 0
    # Cursor was reset.
    assert get_cursor(conn, "zsh") is None
    # Salt preserved.
    assert dedup_salt(conn) == salt_before
    assert get_meta(conn, "dedup_salt_version") == salt_version_before

    # Re-index: rows come back.
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 5


# === 4. New-salt rebuild changes the hash space ===


def test_new_salt_rebuild_changes_hashes(conn, real_dim_embedder) -> None:
    src = _StubSource(
        "zsh",
        [Entry(text="echo hello", ts=100, source="zsh")],
    )
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    hash_before = bytes(conn.execute("SELECT text_hash FROM commands").fetchone()[0])

    # Rotate salt + rebuild + re-index: same raw text → different hash.
    rotate_dedup_salt(conn)
    rebuild(conn)
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    hash_after = bytes(conn.execute("SELECT text_hash FROM commands").fetchone()[0])

    assert hash_before != hash_after, "salt rotation must change the hash"


# === 5. Multi-source dedup via UNIQUE(source, text_hash, ts) ===


def test_same_command_in_two_sources_creates_two_rows(conn, real_dim_embedder) -> None:
    """zsh and bash both record 'ls -la' at ts=100 — UNIQUE allows both
    because (source, text_hash, ts) differs in source."""
    zsh = _StubSource("zsh", [Entry(text="ls -la", ts=100, source="zsh")])
    bash = _StubSource("bash", [Entry(text="ls -la", ts=100, source="bash")])
    index_sources(conn, [zsh, bash], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 2


def test_same_command_same_source_same_ts_dedups(conn, real_dim_embedder) -> None:
    """Two passes of the same source yielding the same entry: row inserted
    once via the indexer's pre-dedup check (UNIQUE constraint backstops)."""
    src = _StubSource("zsh", [Entry(text="ls -la", ts=100, source="zsh")])
    # First pass: cursor was None, entry yielded, inserted.
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 1

    # Second pass: cursor = 100. The stub source skips entries with ts <= since,
    # so it yields nothing; row count unchanged.
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    assert _row_count(conn, "commands") == 1


# === 6. Scrubber integration: secret indexes scrubbed ===


def test_scrubber_redacts_secret_in_indexed_text(conn, real_dim_embedder) -> None:
    src = _StubSource(
        "zsh",
        [
            Entry(
                text="export GITHUB_TOKEN=ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE",
                ts=100,
                source="zsh",
            ),
        ],
    )
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)
    rows = conn.execute("SELECT text_scrubbed FROM commands").fetchall()
    assert len(rows) == 1
    scrubbed = rows[0]["text_scrubbed"]
    assert "ghp_FAKEFAKE" not in scrubbed, "raw GitHub token leaked into commands.text_scrubbed"
    assert "<REDACTED:" in scrubbed


# === 7. Adversarial scrubber-integration test (per Q8 refinement) ===


def test_adversarial_scrubber_corpus_no_pattern_matches_in_db(conn, real_dim_embedder) -> None:
    """Index every line of tests/fixtures/secrets_corpus.txt; assert the
    resulting commands.text_scrubbed contains zero matches against any of
    the scrubber's 18 known secret-pattern types.

    This closes the loop between scrubber coverage and indexer write
    path. The scrubber's own canary tests verify scrub(line) is clean
    in isolation; this test verifies the indexer doesn't somehow let a
    pre-scrubber raw line through into the DB.
    """
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "secrets_corpus.txt"
    lines = [
        ln.strip()
        for ln in fixture.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert len(lines) >= 30, "fixture shrank; expected ≥30 secret-bearing lines"

    src = _StubSource(
        "zsh",
        [Entry(text=ln, ts=1000 + i, source="zsh") for i, ln in enumerate(lines)],
    )
    index_sources(conn, [src], embedder=real_dim_embedder, progress=False)

    rows = conn.execute("SELECT text_scrubbed FROM commands").fetchall()
    indexed_texts = [r["text_scrubbed"] for r in rows]
    combined = "\n".join(indexed_texts)

    # Sentinel check: the canonical FAKEFAKE token in raw inputs should
    # have been replaced by <REDACTED:*> markers. If FAKEFAKE shows up
    # in indexed output, a secret leaked through.
    if "FAKEFAKE" in combined:
        # Find the leaking lines so the failure is debuggable.
        offenders = [t for t in indexed_texts if "FAKEFAKE" in t]
        pytest.fail(
            f"adversarial: {len(offenders)} indexed line(s) still contain "
            f"FAKEFAKE (secret leak through indexer). Sample: {offenders[:3]!r}"
        )

    # Also assert against each of the high-signal scrubber patterns
    # explicitly. These mirror the canary in tests/test_scrub.py and
    # catch regressions where scrub() got bypassed in the indexer.
    pattern_signatures: list[tuple[str, str]] = [
        ("AWS access key", r"AKIA[0-9A-Z]{16}"),
        ("AWS secret-key kwarg", r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{20,}"),
        ("GitHub token (ghp_)", r"\bghp_[A-Za-z0-9]{30,}"),
        ("GitHub token (gho_)", r"\bgho_[A-Za-z0-9]{30,}"),
        ("GitHub token (ghs_)", r"\bghs_[A-Za-z0-9]{30,}"),
        ("GitHub token (ghu_)", r"\bghu_[A-Za-z0-9]{30,}"),
        ("GitHub PAT", r"\bgithub_pat_[A-Za-z0-9_]{30,}"),
        ("OpenAI key", r"\bsk-[A-Za-z0-9]{30,}"),
        ("Slack token", r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
        ("Google API key", r"\bAIzaSy[A-Za-z0-9_-]{30,}"),
        ("URL userinfo (with secret)", r"://[^:/<>\s]+:[^@/<>\s]{6,}@"),
        ("PGPASSWORD env", r"\bPGPASSWORD\s*=\s*[^\s<]{4,}"),
        ("MYSQL_PWD env", r"\bMYSQL_PWD\s*=\s*[^\s<]{4,}"),
        ("--password flag value", r"--password(?:[ =])[^\s<]{4,}"),
        ("mysql -p<value>", r"\bmysql\s+-p[^\s<]{4,}"),
        ("X-API-Key header", r"X-API-Key:\s*[A-Za-z0-9._-]{16,}"),
        ("X-Auth-Token header", r"X-Auth-Token:\s*[A-Za-z0-9._-]{16,}"),
        ("Bearer token", r"Bearer\s+eyJ[A-Za-z0-9._-]{20,}"),
    ]
    for name, pat in pattern_signatures:
        matches = re.findall(pat, combined)
        # The patterns above match secret-shaped strings. Some can still
        # match the <REDACTED:KIND> marker text (e.g. "URL userinfo"
        # might fire if the pattern is too loose). Filter out matches
        # that are entirely composed of redaction markers.
        real_matches = [m for m in matches if "<REDACTED:" not in m]
        assert not real_matches, (
            f"adversarial: scrubber missed {name!r} in indexer write path. "
            f"Matches: {real_matches[:3]!r}"
        )
