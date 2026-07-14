import time
from datetime import UTC, datetime, timedelta

import numpy as np
from conftest import HashEmbedding, QueueOcr

from visual_memory.capture import FrameCandidate
from visual_memory.db import Database
from visual_memory.processing import BackgroundIndexer, EventProcessor, ProcessingItem
from visual_memory.storage import Storage


class FailingOcr:
    name = "failing-ocr"
    available = True
    reason = None

    def recognize(self, frame):
        raise RuntimeError("boom")


def _make_item(session_id="session", seconds=0):
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    candidate = FrameCandidate(
        frame=np.zeros((36, 64, 3), dtype=np.uint8),
        started_at=stamp,
        ended_at=stamp,
        change_score=0.1,
        event_kind="stable",
    )
    return ProcessingItem(session_id=session_id, candidate=candidate)


def _insert_session(db: Database, session_id="session"):
    if not db.fetchone("SELECT id FROM capture_session WHERE id=?", (session_id,)):
        db.execute(
            """INSERT INTO capture_session(id,source_name,started_at,status)
               VALUES(?,?,?,'stopped')""",
            (session_id, "test", datetime(2026, 1, 1, tzinfo=UTC).isoformat()),
        )


def test_store_one_leaves_event_unindexed(settings):
    # store_oneはフレーム保存のみを行い、OCR・埋め込みは行わない(processed_atはNULLのまま)
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    processor = EventProcessor(db, storage, QueueOcr(), HashEmbedding())
    _insert_session(db)

    event_id = processor.store_one(_make_item())

    row = db.fetchone("SELECT ocr_text, processed_at FROM screen_event WHERE id=?", (event_id,))
    assert row["ocr_text"] == ""
    assert row["processed_at"] is None
    evidence = db.fetchone("SELECT id FROM evidence WHERE event_id=? AND kind='frame'", (event_id,))
    assert evidence is not None


def test_background_indexer_processes_pending_events_when_capture_inactive(settings):
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    ocr = QueueOcr(["利益率の推移について"])
    processor = EventProcessor(db, storage, ocr, HashEmbedding())
    _insert_session(db)
    event_id = processor.store_one(_make_item())

    indexer = BackgroundIndexer(
        db, processor, is_capture_active=lambda: False, poll_interval=0.05, idle_interval=0.05
    )
    indexer.start()
    try:
        deadline = time.time() + 5
        row = None
        while time.time() < deadline:
            row = db.fetchone("SELECT ocr_text, processed_at FROM screen_event WHERE id=?", (event_id,))
            if row["processed_at"]:
                break
            time.sleep(0.05)
    finally:
        indexer.stop()

    assert row["processed_at"] is not None
    assert row["ocr_text"] == "利益率の推移について"
    assert indexer.status.indexed_total >= 1
    assert indexer.status.pending_count == 0


def test_background_indexer_pauses_while_capture_is_active(settings):
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    ocr = QueueOcr(["should not appear yet"])
    processor = EventProcessor(db, storage, ocr, HashEmbedding())
    _insert_session(db)
    event_id = processor.store_one(_make_item())

    active = {"value": True}
    indexer = BackgroundIndexer(
        db, processor, is_capture_active=lambda: active["value"], poll_interval=0.05, idle_interval=0.05
    )
    indexer.start()
    try:
        time.sleep(0.3)
        row = db.fetchone("SELECT processed_at FROM screen_event WHERE id=?", (event_id,))
        assert row["processed_at"] is None, "記録中は索引処理が行われてはいけない"
        assert indexer.status.state == "paused-capture"

        active["value"] = False
        indexer.notify()
        deadline = time.time() + 5
        while time.time() < deadline:
            row = db.fetchone("SELECT processed_at FROM screen_event WHERE id=?", (event_id,))
            if row["processed_at"]:
                break
            time.sleep(0.05)
        assert row["processed_at"] is not None
    finally:
        indexer.stop()


def test_background_indexer_gives_up_after_max_attempts(settings):
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    processor = EventProcessor(db, storage, FailingOcr(), HashEmbedding())
    _insert_session(db)
    event_id = processor.store_one(_make_item())

    indexer = BackgroundIndexer(
        db,
        processor,
        is_capture_active=lambda: False,
        poll_interval=0.05,
        idle_interval=0.05,
        max_attempts=2,
    )
    indexer.start()
    try:
        deadline = time.time() + 5
        row = None
        while time.time() < deadline:
            row = db.fetchone("SELECT ocr_text, processed_at FROM screen_event WHERE id=?", (event_id,))
            if row["processed_at"]:
                break
            time.sleep(0.05)
    finally:
        indexer.stop()

    assert row["processed_at"] is not None  # 失敗のまま処理済みとしてマークされ、無限リトライしない
    assert row["ocr_text"] == ""
    assert indexer.status.failures >= 2
