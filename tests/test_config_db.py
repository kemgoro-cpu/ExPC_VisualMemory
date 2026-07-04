from __future__ import annotations

import json
import sqlite3

from visual_memory.config import load_settings
from visual_memory.db import Database


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
