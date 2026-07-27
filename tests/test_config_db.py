from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from visual_memory.config import load_settings
from visual_memory.db import Database

# ocr_text_clean 列が入る前のスキーマ(FTSは ocr_text を索引していた)を再現する。
# 実際の移行では既存ユーザーのDBがこの形なので、そこからの移行を検証する。
_DOWNGRADE_TO_PRE_CLEAN_COLUMN = """
DROP TRIGGER screen_event_ai;
DROP TRIGGER screen_event_ad;
DROP TRIGGER screen_event_au;
DROP TABLE screen_event_fts;
ALTER TABLE screen_event DROP COLUMN ocr_text_clean;
CREATE VIRTUAL TABLE screen_event_fts USING fts5(
    ocr_text, content='screen_event', content_rowid='id',
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
"""


def _make_pre_clean_column_database(path: Path, ocr_text: str) -> Database:
    database = Database(path)
    database.initialize()
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(_DOWNGRADE_TO_PRE_CLEAN_COLUMN)
        connection.commit()
    database.execute(
        "INSERT INTO capture_session(id,source_name,started_at,status) "
        "VALUES('session','test','2026-01-01T00:00:00+00:00','stopped')"
    )
    database.execute(
        """
        INSERT INTO screen_event(
            session_id, started_at, ended_at, frame_path, thumbnail_path, phash,
            width, height, ocr_text, created_at
        ) VALUES('session','2026-01-01T00:00:00+00:00','2026-01-01T00:00:01+00:00',
                 'f.webp','t.webp','0'*16, 1920, 1080, ?, '2026-01-01T00:00:02+00:00')
        """,
        (ocr_text,),
    )
    return database


def _fts_hits(database: Database, query: str) -> int:
    rows = database.fetchall(
        "SELECT rowid FROM screen_event_fts WHERE screen_event_fts MATCH ?", (query,)
    )
    return len(rows)


def test_migration_adds_clean_column_and_keeps_existing_events_searchable(tmp_path):
    # 移行時に ocr_text_clean を生テキストで埋めないと、バックフィルを流すまで
    # 既存イベントが全文検索から消えてしまう
    database = _make_pre_clean_column_database(tmp_path / "migrate.sqlite3", "ECU噴射マップを更新")

    database.initialize()

    row = database.fetchone("SELECT ocr_text, ocr_text_clean FROM screen_event")
    assert row["ocr_text"] == "ECU噴射マップを更新"
    assert row["ocr_text_clean"] == "ECU噴射マップを更新"
    assert _fts_hits(database, "噴射マップ") == 1


def test_migration_is_idempotent(tmp_path):
    database = _make_pre_clean_column_database(tmp_path / "twice.sqlite3", "ECU噴射マップを更新")

    database.initialize()
    database.initialize()

    assert _fts_hits(database, "噴射マップ") == 1


def test_clean_column_update_refreshes_the_full_text_index(tmp_path):
    # バックフィルは ocr_text_clean を更新するだけでFTSが追従する(トリガー任せ)
    database = _make_pre_clean_column_database(
        tmp_path / "trigger.sqlite3", "エクスプローラー\n本文です"
    )
    database.initialize()
    assert _fts_hits(database, "エクスプローラー") == 1

    database.execute("UPDATE screen_event SET ocr_text_clean='本文です'")

    assert _fts_hits(database, "本文です") == 1
    assert _fts_hits(database, "エクスプローラー") == 0


def test_corrupted_and_non_object_config_fall_back_to_defaults(tmp_path, caplog):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = data_dir / "config.json"

    config_path.write_text("{broken", encoding="utf-8")
    settings = load_settings(data_dir)
    assert settings.data_dir == data_dir

    config_path.write_text("[]", encoding="utf-8")
    settings = load_settings(data_dir)
    assert settings.data_dir == data_dir
    assert "falling back to defaults" in caplog.text

    config_path.write_text("{broken again", encoding="utf-8")
    settings.save_region_config(
        [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4}],
        [],
    )
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["ignore_regions"][0]["x"] == 0.1


def test_initialize_closes_its_database_connection(tmp_path, monkeypatch):
    closed = False

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            nonlocal closed
            closed = True
            super().close()

    database = Database(tmp_path / "tracking.sqlite3")
    connection = sqlite3.connect(database.path, factory=TrackingConnection)
    connection.row_factory = sqlite3.Row
    monkeypatch.setattr(database, "connect", lambda: connection)

    database.initialize()

    assert closed is True
