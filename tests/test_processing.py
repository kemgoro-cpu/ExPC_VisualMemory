import threading
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from conftest import HashEmbedding, QueueOcr, distinct_frame

from visual_memory.ai import DisabledOcr, OcrLine, OcrResult
from visual_memory.capture import CaptureManager, ChangeDetector, FrameCandidate, ManualCaptureRequest
from visual_memory.db import Database
from visual_memory.imaging import Region
from visual_memory.processing import (
    BackgroundIndexer,
    ChromeLineTracker,
    EventProcessor,
    ProcessingItem,
)
from visual_memory.storage import Storage


class FailingOcr:
    name = "failing-ocr"
    available = True
    reason = None

    def recognize(self, frame):
        raise RuntimeError("boom")


class PolygonOcr:
    """行ごとの座標を返すOCRの代役。画面上の位置に基づく判定を検証するために使う。"""

    name = "polygon-ocr"
    available = True
    reason = None

    def __init__(self, frames: list[list[tuple[str, float]]]):
        # 1フレーム分を [(テキスト, 縦位置0..1), ...] で与える
        self.frames = list(frames)

    def recognize(self, frame):
        entries = self.frames.pop(0) if self.frames else []
        height = frame.shape[0]
        lines = []
        for text, center_ratio in entries:
            y = center_ratio * height
            lines.append(OcrLine(text=text, confidence=0.9, polygon=[[0, y], [10, y], [10, y], [0, y]]))
        return OcrResult(
            text="\n".join(line.text for line in lines),
            confidence=0.9 if lines else None,
            lines=lines,
        )


class SpyEmbedding(HashEmbedding):
    """encode_document に渡された文字列を記録する埋め込みプロバイダ。"""

    def __init__(self):
        self.documents: list[str] = []

    def encode_document(self, text: str):
        self.documents.append(text)
        return super().encode_document(text)


def _make_item(session_id="session", seconds=0):
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    candidate = FrameCandidate(
        # 記録時の重複スキップに掛からないよう、呼び出しごとに絵柄を変える
        frame=distinct_frame(),
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


def test_process_one_normalizes_ocr_text_before_storing(settings):
    # OCRが誤読(カタカナ混じりの誤字)を返しても、DBには正規化済みのテキストが
    # 保存されること(_index_event内でnormalize_ocr_resultが適用される)
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    ocr = QueueOcr(["テス卜を実行しました"])
    processor = EventProcessor(db, storage, ocr, HashEmbedding())
    _insert_session(db)

    event_id = processor.process_one(_make_item())

    row = db.fetchone("SELECT ocr_text FROM screen_event WHERE id=?", (event_id,))
    assert row["ocr_text"] == "テストを実行しました"


def test_index_event_keeps_raw_text_but_indexes_cleaned_text(settings):
    # 生テキストは証跡として残し、全文検索と埋め込みにはノイズを除いた版を使う
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    ocr = QueueOcr(["14:32\n設計方針をレビューする\nスライド 3/20"])
    embeddings = SpyEmbedding()
    processor = EventProcessor(db, storage, ocr, embeddings)
    _insert_session(db)

    event_id = processor.process_one(_make_item())

    row = db.fetchone("SELECT ocr_text, ocr_text_clean FROM screen_event WHERE id=?", (event_id,))
    assert row["ocr_text"] == "14:32\n設計方針をレビューする\nスライド 3/20"
    assert row["ocr_text_clean"] == "設計方針をレビューする"
    # 埋め込みの入力が洗浄後のテキストであること(生テキストではない)
    assert embeddings.documents == ["設計方針をレビューする"]


def test_index_event_keeps_raw_ocr_lines_in_evidence(settings):
    # 黒塗り判定や監査のため、evidence には除去前の行と座標がそのまま残る
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    processor = EventProcessor(db, storage, QueueOcr(["14:32\n本文です"]), HashEmbedding())
    _insert_session(db)

    event_id = processor.process_one(_make_item())

    evidence = db.fetchone("SELECT text FROM evidence WHERE event_id=? AND kind='ocr'", (event_id,))
    assert evidence["text"] == "14:32\n本文です"


def test_repeated_bottom_line_is_dropped_from_indexed_text(settings):
    # タスクバーのように「毎フレーム画面下端に出る行」は、文字のパターンからは
    # 本文と区別できないため、出現頻度と位置の組み合わせで除去する
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    frames = [[("エクスプローラー", 0.97), (f"検討事項その{index}", 0.5)] for index in range(12)]
    processor = EventProcessor(db, storage, PolygonOcr(frames), HashEmbedding())
    processor.chrome_tracker = ChromeLineTracker(window=300, warmup=5, ratio=0.6)
    _insert_session(db)

    event_ids = [processor.process_one(_make_item(seconds=index)) for index in range(12)]

    last = db.fetchone("SELECT ocr_text, ocr_text_clean FROM screen_event WHERE id=?", (event_ids[-1],))
    assert "エクスプローラー" in last["ocr_text"]  # 生テキストには残る
    assert "エクスプローラー" not in last["ocr_text_clean"]  # 索引用からは消える
    assert "検討事項その11" in last["ocr_text_clean"]  # 本文は残る


def test_chrome_line_tracker_waits_for_warmup_before_judging():
    tracker = ChromeLineTracker(window=100, warmup=5, ratio=0.6)
    for _ in range(4):
        assert tracker.observe("session", {"taskbar"}) == set()
    # 5フレーム目でウォームアップ完了。出現率5/5の行だけが「常時表示」と判定される
    assert tracker.observe("session", {"taskbar", "honbun"}) == {"taskbar"}


def test_chrome_line_tracker_restarts_statistics_on_session_change():
    tracker = ChromeLineTracker(window=100, warmup=2, ratio=0.6)
    tracker.observe("session-a", {"taskbar"})
    assert tracker.observe("session-a", {"taskbar"}) == {"taskbar"}
    # 別セッションでは画面構成が違うため、前セッションの統計は引き継がない
    assert tracker.observe("session-b", {"taskbar"}) == set()


def test_chrome_line_tracker_decays_instead_of_resetting_at_window_limit():
    tracker = ChromeLineTracker(window=4, warmup=2, ratio=0.6)
    for _ in range(4):
        tracker.observe("session", {"taskbar"})
    # 減衰後も判定を継続できること(丸ごとリセットならウォームアップに戻ってしまう)
    assert tracker.observe("session", {"taskbar"}) == {"taskbar"}


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
            row = db.fetchone(
                "SELECT ocr_text, processed_at, indexing_state FROM screen_event WHERE id=?", (event_id,)
            )
            if row["indexing_state"] == "ready":
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


def test_background_indexer_runs_during_capture_when_allowed(settings):
    # GPUワーカー構成(allow_during_capture=True)では、記録中でも索引が進むこと
    db = Database(settings.database_path)
    db.initialize()
    storage = Storage(settings)
    ocr = QueueOcr(["リアルタイム索引"])
    processor = EventProcessor(db, storage, ocr, HashEmbedding())
    _insert_session(db)
    event_id = processor.store_one(_make_item())

    indexer = BackgroundIndexer(
        db,
        processor,
        is_capture_active=lambda: True,  # 記録中のまま
        allow_during_capture=lambda: True,
        poll_interval=0.05,
        idle_interval=0.05,
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
    assert row["ocr_text"] == "リアルタイム索引"


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
            row = db.fetchone(
                "SELECT ocr_text, processed_at, indexing_state FROM screen_event WHERE id=?", (event_id,)
            )
            if row["indexing_state"] == "failed":
                break
            time.sleep(0.05)
    finally:
        indexer.stop()

    assert row["processed_at"] is None
    assert row["indexing_state"] == "failed"
    assert row["ocr_text"] == ""
    assert indexer.status.failures >= 2
    assert indexer._next_pending() is None
    processor.ocr = QueueOcr(["recovered"])
    processor.reprocess_one(event_id)
    assert (
        db.fetchone("SELECT indexing_state FROM screen_event WHERE id=?", (event_id,))["indexing_state"]
        == "ready"
    )


# --- 記録時の重複スキップ ---


def _item_with_frame(frame, session_id="session", seconds=0, kind="stable"):
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return ProcessingItem(
        session_id=session_id,
        candidate=FrameCandidate(
            frame=frame, started_at=stamp, ended_at=stamp, change_score=0.1, event_kind=kind
        ),
    )


def _processor(settings):
    db = Database(settings.database_path)
    db.initialize()
    processor = EventProcessor(db, Storage(settings), QueueOcr(), HashEmbedding())
    _insert_session(db)
    return db, processor


def test_same_screen_is_not_stored_twice(settings):
    # マウスカーソルが動いただけの画面は、変化検出には引っ掛かるが内容は同じ。
    # 保存段階で落として履歴と容量を無駄にしない
    db, processor = _processor(settings)
    frame = distinct_frame(30)

    first = processor.store_one(_item_with_frame(frame, seconds=0))
    second = processor.store_one(_item_with_frame(frame.copy(), seconds=5))

    assert first is not None
    assert second is None
    assert processor.status.skipped_duplicates == 1
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 1


def test_skipped_duplicate_extends_the_previous_event_end_time(settings):
    # 画像は捨てても「その画面が映っていた時間」は残す
    db, processor = _processor(settings)
    frame = distinct_frame(30)

    event_id = processor.store_one(_item_with_frame(frame, seconds=0))
    processor.store_one(_item_with_frame(frame.copy(), seconds=45))

    row = db.fetchone("SELECT started_at, ended_at FROM screen_event WHERE id=?", (event_id,))
    assert row["started_at"] != row["ended_at"]
    assert row["ended_at"].startswith("2026-01-01T00:00:45")


def test_manual_capture_is_stored_even_when_the_screen_is_unchanged(settings):
    # ユーザーが明示的に押した操作は、同じ画面でも必ず残す
    db, processor = _processor(settings)
    frame = distinct_frame(30)

    processor.store_one(_item_with_frame(frame, seconds=0))
    manual = processor.store_one(_item_with_frame(frame.copy(), seconds=5, kind="manual"))

    assert manual is not None
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 2


def test_a_small_content_change_is_still_stored(settings):
    # 取りこぼし防止の確認。文字が少し変わった程度でも別の画面として残す
    db, processor = _processor(settings)
    frame = distinct_frame(30)
    changed = frame.copy()
    changed[20:30, 10:50] = 255  # 本文が書き換わった程度の変化

    processor.store_one(_item_with_frame(frame, seconds=0))
    second = processor.store_one(_item_with_frame(changed, seconds=5))

    assert second is not None
    assert processor.status.skipped_duplicates == 0
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 2


def test_watch_region_small_change_survives_storage_dedup(settings):
    settings.watch_regions = [{"x": 0.4, "y": 0.4, "width": 0.1, "height": 0.1}]
    db, processor = _processor(settings)
    detector = ChangeDetector(watch_regions=[Region(0.4, 0.4, 0.1, 0.1)])
    base = datetime(2026, 1, 1, tzinfo=UTC)
    before = np.random.default_rng(1).integers(0, 200, (100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[44:46, 44:46] = 255
    initial = detector.push(before, 0, base)[0]
    processor.store_one(ProcessingItem("session", initial))
    detector.push(after, 1, base + timedelta(seconds=1))
    changed = detector.push(after, 3, base + timedelta(seconds=3))[0]
    assert processor.store_one(ProcessingItem("session", changed)) is not None
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 2


def test_deleted_latest_event_does_not_suppress_next_heartbeat(settings):
    db, processor = _processor(settings)
    frame = distinct_frame(40)
    deleted = processor.store_one(_item_with_frame(frame))
    db.execute("DELETE FROM screen_event WHERE id=?", (deleted,))
    replacement = processor.store_one(_item_with_frame(frame, seconds=60, kind="heartbeat"))
    assert replacement is not None and replacement != deleted
    assert db.fetchone("SELECT id FROM screen_event")["id"] == replacement


def test_storage_metrics_exclude_duplicates_and_reindexing(settings):
    _, processor = _processor(settings)
    frame = distinct_frame(40)
    event_id = processor.store_one(_item_with_frame(frame))
    saved_at = processor.status.last_saved_at
    processor.store_one(_item_with_frame(frame, seconds=60, kind="heartbeat"))
    processor.reprocess_one(event_id)
    assert processor.status.saved_count == processor.status.processed == 1
    assert processor.status.session_saved_count == 1
    assert processor.status.last_saved_at == saved_at


def test_manual_storage_result_is_idempotent_and_has_durable_evidence(settings):
    db, processor = _processor(settings)
    request = ManualCaptureRequest("manual-once")
    item = _item_with_frame(distinct_frame(), kind="manual")
    item.candidate.manual_request = request
    event_id = processor.store_one(item)
    assert request.snapshot()["state"] == "saved"
    row = db.fetchone("SELECT * FROM screen_event WHERE id=?", (event_id,))
    with processor.storage.open_image(row["frame_path"]) as image:
        image.verify()
    with processor.storage.open_image(row["thumbnail_path"]) as image:
        image.verify()
    assert db.fetchone("SELECT id FROM evidence WHERE event_id=? AND kind='frame'", (event_id,))
    retry = ManualCaptureRequest("manual-once")
    item.candidate.manual_request = retry
    assert processor.store_one(item) == event_id
    assert retry.snapshot()["state"] == "saved"
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 1
    assert processor.status.saved_count == 1


def test_manual_failed_transaction_does_not_report_success(settings):
    db, processor = _processor(settings)
    db.execute(
        "CREATE TRIGGER reject_frame BEFORE INSERT ON evidence "
        "BEGIN SELECT RAISE(ABORT,'fixture failure'); END"
    )
    item = _item_with_frame(distinct_frame(), kind="manual")
    request = item.candidate.manual_request = ManualCaptureRequest("failed-commit")
    with pytest.raises(Exception, match="fixture failure"):
        processor.store_one(item)
    assert request.snapshot()["state"] == "failed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 0
    assert processor.status.saved_count == 0
    assert list(settings.frames_dir.rglob("*.webp")) == []


def test_manual_queue_eviction_and_closed_queue_report_failure(settings):
    db, processor = _processor(settings)
    processor.queue.maxsize = 1
    first = _item_with_frame(distinct_frame(), kind="manual")
    first.candidate.manual_request = ManualCaptureRequest("evicted")
    processor.submit("session", first.candidate)
    latest = _item_with_frame(distinct_frame(), seconds=1, kind="manual")
    latest.candidate.manual_request = ManualCaptureRequest("latest")
    processor.submit("session", latest.candidate)
    assert first.candidate.manual_request.snapshot()["state"] == "failed"
    processor.store_one(processor.queue.get())
    assert latest.candidate.manual_request.snapshot()["state"] == "saved"
    processor.queue.close()
    closed = _item_with_frame(distinct_frame(), seconds=2, kind="manual")
    closed.candidate.manual_request = ManualCaptureRequest("closed")
    assert not processor.submit("session", closed.candidate)
    assert closed.candidate.manual_request.snapshot()["state"] == "failed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 1


def test_manual_capture_at_capacity_returns_failure(settings):
    db, processor = _processor(settings)
    settings.max_storage_bytes = 0
    ready = threading.Event()

    class Source:
        def frames(self, stop):
            while not stop.is_set():
                yield np.zeros((36, 64, 3), dtype=np.uint8)
                ready.set()
                stop.wait(0.02)

        def close(self):
            pass

    manager = CaptureManager(settings, processor.submit, lambda *a: None, lambda *a: None, lambda _: Source())
    processor.start()
    manager.start("test")
    try:
        assert ready.wait(2)
        result = manager.request_manual_capture(request_id="capacity", timeout=2)
        assert result["state"] == "failed"
        assert result["event_id"] is None
        assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 0
    finally:
        manager.stop()
        processor.stop()


def test_disabled_ocr_remains_blocked_then_recovers_automatically(settings):
    db, processor = _processor(settings)
    processor.ocr = DisabledOcr("offline")
    event_id = processor.store_one(_make_item())
    indexer = BackgroundIndexer(db, processor, lambda: False, poll_interval=0.02, idle_interval=0.02)
    indexer._process_row(indexer._next_pending())
    blocked = db.fetchone("SELECT * FROM screen_event WHERE id=?", (event_id,))
    assert blocked["processed_at"] is None
    assert blocked["indexing_state"] == "blocked"
    assert blocked["index_attempts"] == 0
    assert indexer.status.indexed_total == 0
    processor.ocr = QueueOcr(["recovered searchable"])
    indexer.start()
    try:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            row = db.fetchone("SELECT * FROM screen_event WHERE id=?", (event_id,))
            if row["indexing_state"] == "ready":
                break
            time.sleep(0.01)
        assert row["indexing_state"] == "ready"
        assert row["ocr_text"] == "recovered searchable"
    finally:
        indexer.stop()


def test_normal_empty_ocr_is_ready_not_failed(settings):
    db, processor = _processor(settings)
    event_id = processor.process_one(_make_item())
    row = db.fetchone("SELECT * FROM screen_event WHERE id=?", (event_id,))
    assert row["ocr_text"] == ""
    assert row["indexing_state"] == "ready"
    assert row["processed_at"] is not None
    assert row["indexing_error"] is None


def test_image_write_failure_finishes_request_without_saved_metrics(settings, monkeypatch):
    db, processor = _processor(settings)
    item = _item_with_frame(distinct_frame(), kind="manual")
    request = item.candidate.manual_request = ManualCaptureRequest("write-error")

    def fail_write(*args):
        raise OSError("fixture disk error")

    monkeypatch.setattr(processor.storage, "save_frame", fail_write)
    with pytest.raises(OSError, match="fixture disk error"):
        processor.store_one(item)
    assert request.snapshot()["state"] == "failed"
    assert db.fetchone("SELECT COUNT(*) AS n FROM screen_event")["n"] == 0
    assert processor.status.saved_count == 0


def test_stop_without_worker_fails_pending_request(settings):
    _, processor = _processor(settings)
    item = _item_with_frame(distinct_frame(), kind="manual")
    request = item.candidate.manual_request = ManualCaptureRequest("never-started")
    processor.submit("session", item.candidate)
    processor.stop()
    assert request.snapshot()["state"] == "failed"
