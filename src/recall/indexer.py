"""Indexer: source → scrubber → embedder → DB write.

The orchestration layer. Pulls entries from one or more
``HistorySource`` instances, runs each through the scrubber, hashes
the scrubbed text under the dedup salt, and writes
(text_scrubbed, text_hash, vector, metadata) to ``commands`` +
``commands_vec`` in batched transactions.

ARCHITECTURAL INVARIANT (LOCKED Q1, CLAUDE.md §1)

    Everything below the scrubber is the scrubbed text. Sources yield
    raw text; the indexer scrubs once at entry, then hashes the
    scrubbed form, embeds the scrubbed form, and stores the scrubbed
    form. Raw text never leaves the indexer's process boundary.
    Two raw commands that scrub to the same scrubbed text dedup to
    one row — arguably correct semantics (same intent post-scrub).

ARCHITECTURAL SEAM (CLAUDE.md "Architectural seams")

    source (streaming) → indexer (chunks) → embedder (batches) → DB write (per-batch)

    Sources stream via iter_entries — never materialized via
    list(...) which would defeat the streaming advantage. The indexer
    chunks streaming output into INDEXER_BATCH_ROWS-sized buffers,
    sends each buffer to the embedder (whose internal batch_size is
    a separate, smaller knob), and commits each buffer as one DB
    transaction. Memory bounded by INDEXER_BATCH_ROWS.

INCREMENTAL CURSOR (LOCKED Q2)

    Per-source ts cursor in meta (key ``cursor_<source>``). Each
    source's iter_entries(since=cursor) resumes after the last
    indexed entry. The cursor advances atomically with the row
    inserts it represents (within the same transaction). On crash
    mid-batch, the cursor stays at the last successfully committed
    batch's max ts — restart re-processes only those entries that
    weren't committed.

REBUILD SEMANTICS (LOCKED Q5, CLAUDE.md §4b)

    --rebuild: DROP + CREATE commands + commands_vec + commands_fts
        (the well-tested vec0 reset path; salt preserved).
    --rebuild --new-salt: rotates salt then rebuilds.
    --new-salt without --rebuild: rejected by the CLI (the salt and
        rows must agree on hash provenance).

    Cursor metas are also reset on --rebuild — re-indexing from
    scratch starts cursors at zero so all entries are processed.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from recall.db import (
    dedup_hash,
    dedup_salt,
    get_cursor,
    set_cursor,
    set_meta,
    transaction,
)
from recall.embed import Embedder
from recall.scrub import scrub
from recall.sources.base import Entry, HistorySource

_LOG = logging.getLogger(__name__)

# Indexer-batch size — how many entries we buffer between DB transactions.
# Locked at 1024 per Q4: the embedder's internal batch_size (default 64)
# handles GPU/CPU memory throughput; the indexer batch handles transaction
# amortization. Memory: 1024 × 384 × 4 = 1.5 MB per buffer.
INDEXER_BATCH_ROWS = 1024

# Progress-line cadence (seconds). Per Q7: periodic stderr line, no tqdm.
PROGRESS_INTERVAL_S = 2.5


@dataclass
class IndexResult:
    """Summary of one ``index_sources`` invocation."""

    inserted: int = 0
    skipped_dedup: int = 0  # row already present (UNIQUE conflict)
    skipped_scrub_only: int = 0  # entry's scrubbed text matched an existing hash
    by_source: dict[str, int] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    embedder_seconds: float = 0.0
    db_write_seconds: float = 0.0

    def total_processed(self) -> int:
        return self.inserted + self.skipped_dedup + self.skipped_scrub_only


def _drop_and_create(conn: sqlite3.Connection) -> None:
    """Rebuild path: drop and re-create commands + commands_vec + commands_fts.

    sqlite-vec's vec0 virtual table is well-tested on CREATE-from-scratch
    but historically not on DELETE FROM (locked Q5). DROP + CREATE is the
    unambiguous reset. Salt and other meta keys are preserved across this
    operation — only the row stores reset.

    The FTS5 sync triggers reference both ``commands`` and ``commands_fts``,
    so they're dropped first. Recreated from migrations/0001_initial.sql.
    """
    from importlib.resources import files

    # Order matters: drop triggers (which reference commands + commands_fts)
    # before dropping their target tables. Then drop indices; then drop the
    # FTS and vec virtual tables; then drop commands. Finally, re-run the
    # original schema's CREATE statements (filtered to only the parts we
    # dropped — meta stays intact).
    drop_sql = """
        DROP TRIGGER IF EXISTS commands_ai;
        DROP TRIGGER IF EXISTS commands_ad;
        DROP TRIGGER IF EXISTS commands_au;
        DROP TABLE   IF EXISTS commands_fts;
        DROP TABLE   IF EXISTS commands_vec;
        DROP INDEX   IF EXISTS idx_commands_session;
        DROP INDEX   IF EXISTS idx_commands_cwd;
        DROP INDEX   IF EXISTS idx_commands_ts;
        DROP TABLE   IF EXISTS commands;
    """
    conn.executescript(drop_sql)

    # Pull the original migration text and re-execute only the row-store
    # CREATE block (everything except CREATE TABLE meta and the meta seed).
    full = (files("recall.migrations") / "0001_initial.sql").read_text(encoding="utf-8")
    # Heuristic: skip lines that are part of the meta CREATE/INSERT block
    # and execute the rest. The migration file's structure is stable
    # (CLAUDE.md §1.3 — never edit post-release) so this string match is safe.
    create_block = full[full.index("CREATE TABLE commands") :]
    conn.executescript(create_block)

    # Reset cursor metas — fresh rebuild means all sources re-index from zero.
    # Use a SELECT then DELETE pattern so we touch only cursor keys.
    keys = [
        row["key"]
        for row in conn.execute("SELECT key FROM meta WHERE key LIKE 'cursor_%'").fetchall()
    ]
    for k in keys:
        conn.execute("DELETE FROM meta WHERE key = ?", (k,))


def rebuild(conn: sqlite3.Connection) -> None:
    """Public entry point for the --rebuild path.

    Does NOT wrap _drop_and_create in transaction() — _drop_and_create
    uses ``executescript()``, which issues an implicit COMMIT before
    running, leaving no transaction for the wrapper to commit (CLAUDE.md
    "Composition is where bugs live", instance #2 — the exact same trap
    1.3's migrate() fell into). SQLite DDL is durable per-statement; the
    sequence of DROP/CREATE statements is robust against mid-rebuild
    interruption (next rebuild() finds whatever subset of tables exist
    and DROPs IF EXISTS).

    Caller (the CLI) is responsible for calling rotate_dedup_salt before
    rebuild() if --new-salt was also requested. See CLAUDE.md §4b.
    """
    _drop_and_create(conn)
    _LOG.info("recall.indexer: rebuilt commands/commands_vec/commands_fts; cursors cleared")


def _process_batch(
    conn: sqlite3.Connection,
    batch: list[Entry],
    embedder: Embedder,
    salt: bytes,
    timing: IndexResult,
) -> tuple[int, int]:
    """Scrub, hash, dedup, embed, and write one batch.

    Returns ``(inserted, skipped)``. The caller advances cursors based
    on the batch's max-ts AFTER this returns successfully.

    ARCHITECTURAL INVARIANT: scrubbing happens FIRST. The hash and the
    embedding both consume the scrubbed text. Two raw entries that
    scrub to the same text collapse to a single hash; the second
    insert hits the UNIQUE constraint and is reported as a dedup
    skip.
    """
    if not batch:
        return 0, 0

    # Step 1: scrub each entry's raw text. This is where raw text dies.
    scrubbed_texts: list[str] = [scrub(e.text) for e in batch]

    # Step 2: compute hash over scrubbed text. Salt is the current
    # meta.dedup_salt at indexer construction time.
    hashes: list[bytes] = [dedup_hash(salt, t) for t in scrubbed_texts]

    # Step 3: pre-dedup against the DB so we don't waste embedder
    # cycles on rows that will hit UNIQUE conflicts. Query existing
    # (source, text_hash, ts) triples for the batch's hashes.
    # SQLite has a 999-parameter default limit; we'd need 3 × 1024 =
    # 3072 to query all keys at once via IN clause. Chunk to be safe.
    keys_to_check = [(e.source, h, e.ts) for e, h in zip(batch, hashes, strict=True)]
    existing: set[tuple[str, bytes, int]] = set()
    for chunk_start in range(0, len(keys_to_check), 200):
        chunk = keys_to_check[chunk_start : chunk_start + 200]
        params: list[object] = []
        for s, h, t in chunk:
            params.extend([s, h, t])
        # SQLite doesn't support row-value IN with mixed-type tuples
        # cleanly, so fall back to a per-chunk SELECT with OR.
        where = " OR ".join(["(source = ? AND text_hash = ? AND ts = ?)"] * len(chunk))
        rows = conn.execute(
            f"SELECT source, text_hash, ts FROM commands WHERE {where}",
            params,
        ).fetchall()
        for r in rows:
            existing.add((str(r["source"]), bytes(r["text_hash"]), int(r["ts"])))

    # Step 4: filter the batch to cache-misses only. These get embedded.
    miss_indices: list[int] = []
    for i, (e, h) in enumerate(zip(batch, hashes, strict=True)):
        if (e.source, h, e.ts) in existing:
            continue
        miss_indices.append(i)

    skipped = len(batch) - len(miss_indices)

    if not miss_indices:
        return 0, skipped

    # Step 5: embed cache-miss scrubbed texts. The embedder handles
    # internal batching at its own batch_size; we just hand it the
    # full miss list.
    miss_texts = [scrubbed_texts[i] for i in miss_indices]
    t0 = time.perf_counter()
    miss_vectors = embedder.encode(miss_texts)
    timing.embedder_seconds += time.perf_counter() - t0

    # Step 6: write to DB in one transaction. INSERT into commands +
    # INSERT into commands_vec for each miss. Cursor advance happens
    # in the caller AFTER this commits successfully.
    t0 = time.perf_counter()
    with transaction(conn):
        inserted = 0
        for j, idx in enumerate(miss_indices):
            entry = batch[idx]
            scrubbed = scrubbed_texts[idx]
            h = hashes[idx]
            vec = miss_vectors[j]
            try:
                cur = conn.execute(
                    """
                    INSERT INTO commands (
                        source, source_id, text_scrubbed, text_hash,
                        cwd, hostname, exit_code, duration_ms, session_id, ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        entry.source,
                        entry.source_id,
                        scrubbed,
                        h,
                        entry.cwd,
                        entry.hostname,
                        entry.exit_code,
                        entry.duration_ms,
                        entry.session_id,
                        entry.ts,
                    ),
                )
                command_id = cur.lastrowid
                conn.execute(
                    "INSERT INTO commands_vec (command_id, embedding) VALUES (?, ?)",
                    (command_id, vec.tobytes()),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # Race with a parallel pre-dedup miss (shouldn't happen
                # in our sequential indexer; defensive).
                skipped += 1
    timing.db_write_seconds += time.perf_counter() - t0

    return inserted, skipped


def _stream_with_cursor(
    source: HistorySource,
    conn: sqlite3.Connection,
) -> Iterator[Entry]:
    """Yield entries from ``source`` starting after that source's cursor.

    Cursor is read from ``meta.cursor_<source.name>`` (None == start
    from the beginning).
    """
    cursor = get_cursor(conn, source.name)
    yield from source.iter_entries(since=cursor)


def index_sources(
    conn: sqlite3.Connection,
    sources: Sequence[HistorySource],
    embedder: Embedder | None = None,
    *,
    progress: bool = True,
) -> IndexResult:
    """Run one indexing pass over ``sources``, sequentially (Q3 lock).

    Each source is fully drained before the next starts. Within a source,
    entries are buffered into INDEXER_BATCH_ROWS-size batches; each batch
    triggers one embedder.encode() call and one DB transaction. Cursor
    advances after each successful batch commit, so a crash mid-source
    leaves the cursor at the last committed batch's max ts.

    Per Q3: sequential (not parallel) because the embedder is already
    batched internally — threading the orchestration adds GIL contention
    without real parallelism.

    Per Q7: progress is a periodic stderr line (every PROGRESS_INTERVAL_S
    seconds), not tqdm. ``progress=False`` disables it entirely (used
    by tests).
    """
    if embedder is None:
        embedder = Embedder()

    salt = dedup_salt(conn)
    result = IndexResult()
    overall_start = time.perf_counter()
    last_progress = overall_start

    def _flush(
        batch: list[Entry],
        max_ts: int,
        source_name: str,
    ) -> tuple[int, int]:
        # Module-scope-style flush: takes source_name explicitly to avoid
        # closure-over-loop-variable late-binding (B023). The result and
        # salt and embedder come from the enclosing index_sources scope,
        # which is fixed across the call.
        ins, skip = _process_batch(conn, batch, embedder, salt, result)
        if max_ts > 0:
            # Advance cursor only when a batch had a known timestamp.
            # Entries with ts=0 (unknown) leave the cursor untouched.
            with transaction(conn):
                set_cursor(conn, source_name, max_ts)
        return ins, skip

    for source in sources:
        source_inserted = 0
        batch: list[Entry] = []
        max_ts_in_batch = 0

        for entry in _stream_with_cursor(source, conn):
            batch.append(entry)
            if entry.ts > max_ts_in_batch:
                max_ts_in_batch = entry.ts
            if len(batch) >= INDEXER_BATCH_ROWS:
                ins, skip = _flush(batch, max_ts_in_batch, source.name)
                source_inserted += ins
                result.inserted += ins
                result.skipped_dedup += skip
                batch = []
                max_ts_in_batch = 0

                # Periodic progress line.
                now = time.perf_counter()
                if progress and (now - last_progress) >= PROGRESS_INTERVAL_S:
                    elapsed = now - overall_start
                    print(
                        f"recall index: {source.name} +{source_inserted} (~{elapsed:.0f}s elapsed)",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_progress = now

        # Flush trailing partial batch.
        if batch:
            ins, skip = _flush(batch, max_ts_in_batch, source.name)
            source_inserted += ins
            result.inserted += ins
            result.skipped_dedup += skip

        result.by_source[source.name] = source_inserted
        if progress:
            elapsed = time.perf_counter() - overall_start
            print(
                f"recall index: {source.name} done +{source_inserted} (~{elapsed:.0f}s elapsed)",
                file=sys.stderr,
                flush=True,
            )

    # Stamp the embedder model into meta so we can detect mismatches at
    # query time (CLAUDE.md §4 "Embedding consistency"). Idempotent.
    set_meta(conn, "embedding_model_name", embedder.model_name)
    set_meta(conn, "embedding_model_revision", embedder.model_revision or "")

    result.runtime_seconds = time.perf_counter() - overall_start
    return result


__all__ = (
    "INDEXER_BATCH_ROWS",
    "IndexResult",
    "index_sources",
    "rebuild",
)
