from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS capture_session (
    id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    error TEXT
);

CREATE TABLE IF NOT EXISTS screen_event (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES capture_session(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    ended_at TEXT NOT NULL,
    frame_path TEXT NOT NULL,
    thumbnail_path TEXT NOT NULL,
    phash TEXT NOT NULL,
    change_score REAL NOT NULL DEFAULT 0,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    event_kind TEXT NOT NULL DEFAULT 'stable',
    ocr_text TEXT NOT NULL DEFAULT '',
    ocr_confidence REAL,
    embedding BLOB,
    embedding_dim INTEGER,
    previous_event_id INTEGER REFERENCES screen_event(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    processed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_screen_event_time ON screen_event(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_screen_event_session ON screen_event(session_id, started_at);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES screen_event(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('frame', 'ocr', 'note', 'file', 'clipboard')),
    path TEXT,
    text TEXT,
    mime_type TEXT,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_evidence_event ON evidence(event_id, kind);

CREATE TABLE IF NOT EXISTS context_pack (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT '',
    deduplicate_overlaps INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL CHECK(status IN ('draft', 'approved', 'revoked', 'expired')),
    approved_at TEXT,
    expires_at TEXT,
    revoked_at TEXT,
    manifest_sha256 TEXT,
    artifact_path TEXT,
    build_error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_pack_item (
    pack_id TEXT NOT NULL REFERENCES context_pack(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES screen_event(id) ON DELETE RESTRICT,
    position INTEGER NOT NULL,
    image_path TEXT,
    ocr_text TEXT NOT NULL DEFAULT '',
    redactions_json TEXT NOT NULL DEFAULT '[]',
    sha256 TEXT,
    PRIMARY KEY(pack_id, event_id),
    UNIQUE(pack_id, position)
);

CREATE INDEX IF NOT EXISTS idx_context_pack_status ON context_pack(status, expires_at);

CREATE VIRTUAL TABLE IF NOT EXISTS screen_event_fts USING fts5(
    ocr_text,
    content='screen_event',
    content_rowid='id',
    tokenize='trigram case_sensitive 0'
);

CREATE TRIGGER IF NOT EXISTS screen_event_ai AFTER INSERT ON screen_event BEGIN
  INSERT INTO screen_event_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS screen_event_ad AFTER DELETE ON screen_event BEGIN
  INSERT INTO screen_event_fts(screen_event_fts, rowid, ocr_text)
  VALUES('delete', old.id, old.ocr_text);
END;
CREATE TRIGGER IF NOT EXISTS screen_event_au AFTER UPDATE OF ocr_text ON screen_event BEGIN
  INSERT INTO screen_event_fts(screen_event_fts, rowid, ocr_text)
  VALUES('delete', old.id, old.ocr_text);
  INSERT INTO screen_event_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
END;
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            pack_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(context_pack)").fetchall()
            }
            if "deduplicate_overlaps" not in pack_columns:
                # Existing documents retain their original one-frame-per-page behavior.
                connection.execute(
                    "ALTER TABLE context_pack ADD COLUMN "
                    "deduplicate_overlaps INTEGER NOT NULL DEFAULT 0"
                )
            if "artifact_path" not in pack_columns:
                connection.execute("ALTER TABLE context_pack ADD COLUMN artifact_path TEXT")
            if "build_error" not in pack_columns:
                connection.execute("ALTER TABLE context_pack ADD COLUMN build_error TEXT")
            fts = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='screen_event_fts'"
            ).fetchone()
            if fts and "trigram" not in str(fts["sql"] or "").lower():
                connection.executescript(
                    """
                    DROP TRIGGER IF EXISTS screen_event_ai;
                    DROP TRIGGER IF EXISTS screen_event_ad;
                    DROP TRIGGER IF EXISTS screen_event_au;
                    DROP TABLE screen_event_fts;
                    CREATE VIRTUAL TABLE screen_event_fts USING fts5(
                        ocr_text,
                        content='screen_event',
                        content_rowid='id',
                        tokenize='trigram case_sensitive 0'
                    );
                    CREATE TRIGGER screen_event_ai AFTER INSERT ON screen_event BEGIN
                      INSERT INTO screen_event_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
                    END;
                    CREATE TRIGGER screen_event_ad AFTER DELETE ON screen_event BEGIN
                      INSERT INTO screen_event_fts(screen_event_fts, rowid, ocr_text)
                      VALUES('delete', old.id, old.ocr_text);
                    END;
                    CREATE TRIGGER screen_event_au AFTER UPDATE OF ocr_text ON screen_event BEGIN
                      INSERT INTO screen_event_fts(screen_event_fts, rowid, ocr_text)
                      VALUES('delete', old.id, old.ocr_text);
                      INSERT INTO screen_event_fts(rowid, ocr_text) VALUES (new.id, new.ocr_text);
                    END;
                    INSERT INTO screen_event_fts(screen_event_fts) VALUES('rebuild');
                    """
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, params)
            connection.commit()
            return int(cursor.lastrowid or 0)

    def executemany(self, sql: str, params: Sequence[Sequence[Any]]) -> None:
        with self.connect() as connection:
            connection.executemany(sql, params)
            connection.commit()

    def fetchone(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute(sql, params).fetchall())
