-- Recall schema v1 (Commit 1.3).
--
-- Tables:
--   meta           — key/value config (salt, scrubber version, embedding model)
--   commands       — scrubbed shell-history entries with metadata
--   commands_vec   — sqlite-vec virtual table holding embeddings (dim 384)
--   commands_fts   — FTS5 virtual table for lexical search, content-linked
--                    to `commands` via content='commands' content_rowid='id'
--
-- Hard rules:
--   - `text_scrubbed` is always the post-scrubber text. Raw history text
--     never lands in this DB.
--   - `text_hash` = BLAKE2b(meta.dedup_salt ‖ raw_text), digest_size=32.
--     The salt that was current at insertion time is the only one that
--     can reproduce the hash; see CLAUDE.md section 4a for rotation rules.
--   - `commands_vec.embedding` dimension MUST match the model recorded in
--     meta.embedding_model_*.
--   - All bumps to this schema land as a new migrations/000N_*.sql file
--     plus a branch in db.migrate(); never edit this file post-release.

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE commands (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT    NOT NULL,            -- 'zsh' | 'bash' | 'fish' | 'atuin'
    source_id     TEXT,                        -- atuin row id, NULL for histfile sources
    text_scrubbed TEXT    NOT NULL,
    text_hash     BLOB    NOT NULL,            -- BLAKE2b(salt ‖ raw), 32 bytes
    cwd           TEXT,
    hostname      TEXT,
    exit_code     INTEGER,
    duration_ms   INTEGER,
    session_id    TEXT,
    ts            INTEGER NOT NULL,            -- unix seconds
    UNIQUE(source, text_hash, ts)
);

CREATE INDEX idx_commands_ts      ON commands(ts DESC);
CREATE INDEX idx_commands_cwd     ON commands(cwd);
CREATE INDEX idx_commands_session ON commands(session_id, ts);

-- Embeddings live in a sqlite-vec virtual table. Dimension 384 matches
-- bge-small-en-v1.5 (the model name pinned in meta).
CREATE VIRTUAL TABLE commands_vec USING vec0(
    command_id INTEGER PRIMARY KEY,
    embedding  FLOAT[384]
);

-- FTS5 lexical fallback, content-linked to commands.
CREATE VIRTUAL TABLE commands_fts USING fts5(
    text_scrubbed,
    content='commands',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 1'
);

-- FTS5 sync triggers (standard external-content idiom).
CREATE TRIGGER commands_ai AFTER INSERT ON commands BEGIN
    INSERT INTO commands_fts(rowid, text_scrubbed)
        VALUES (new.id, new.text_scrubbed);
END;

CREATE TRIGGER commands_ad AFTER DELETE ON commands BEGIN
    INSERT INTO commands_fts(commands_fts, rowid, text_scrubbed)
        VALUES ('delete', old.id, old.text_scrubbed);
END;

CREATE TRIGGER commands_au AFTER UPDATE ON commands BEGIN
    INSERT INTO commands_fts(commands_fts, rowid, text_scrubbed)
        VALUES ('delete', old.id, old.text_scrubbed);
    INSERT INTO commands_fts(rowid, text_scrubbed)
        VALUES (new.id, new.text_scrubbed);
END;
