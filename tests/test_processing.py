import time
from datetime import UTC, datetime, timedelta

import numpy as np
from conftest import HashEmbedding, QueueOcr

from visual_memory.ai import OcrLine, OcrResult
from visual_memory.capture import FrameCandidate
from visual_memory.db import Database
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
            lines.append(
                OcrLine(text=text, confidence=0.9, polygon=[[0, y], [10, y], [10, y], [0, y]])
            )
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

    row = db.fetchone(
        "SELECT ocr_text, ocr_text_clean FROM screen_event WHERE id=?", (event_id,)
    )
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

    evidence = db.fetchone(
        "SELECT text FROM evidence WHERE event_id=? AND kind='ocr'", (event_id,)
    )
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

    last = db.fetchone(
        "SELECT ocr_text, ocr_text_clean FROM screen_event WHERE id=?", (event_ids[-1],)
    )
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
            row = db.fetchone("SELECT ocr_text, processed_at FROM screen_event WHERE id=?", (event_id,))
            if row["processed_at"]:
                break
            time.sleep(0.05)
    finally:
        indexer.stop()

    assert row["processed_at"] is not None  # 失敗のまま処理済みとしてマークされ、無限リトライしない
    assert row["ocr_text"] == ""
    assert indexer.status.failures >= 2
