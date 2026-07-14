from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from .ai import EmbeddingProvider, OcrProvider
from .capture import FrameCandidate, iso
from .db import Database
from .imaging import perceptual_hash, to_bgr
from .storage import Storage
from .textnorm import normalize_ocr_result

LOGGER = logging.getLogger(__name__)


class LatestQueue[T]:
    """Bounded queue that retains the newest visual state when processing falls behind."""

    def __init__(self, maxsize: int = 8):
        self.maxsize = maxsize
        self._items: deque[T] = deque()
        self._condition = threading.Condition()
        self.closed = False
        self.drops = 0

    def put(self, item: T) -> bool:
        dropped = False
        with self._condition:
            if self.closed:
                return False
            if len(self._items) >= self.maxsize:
                self._items.popleft()
                self.drops += 1
                dropped = True
            self._items.append(item)
            self._condition.notify()
        return not dropped

    def get(self, timeout: float | None = None) -> T | None:
        end = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items and not self.closed:
                remaining = None if end is None else max(0.0, end - time.monotonic())
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            return self._items.popleft() if self._items else None

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


@dataclass(slots=True)
class ProcessingItem:
    session_id: str
    candidate: FrameCandidate


@dataclass(slots=True)
class ProcessorStatus:
    state: str = "stopped"
    queue_depth: int = 0
    queue_drops: int = 0
    processed: int = 0
    failures: int = 0
    last_error: str | None = None
    last_processed_at: str | None = None


class EventProcessor:
    def __init__(
        self,
        db: Database,
        storage: Storage,
        ocr: OcrProvider,
        embeddings: EmbeddingProvider,
        queue_size: int = 8,
    ):
        self.db = db
        self.storage = storage
        self.ocr = ocr
        self.embeddings = embeddings
        self.queue: LatestQueue[ProcessingItem] = LatestQueue(queue_size)
        self.status = ProcessorStatus()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._index_lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.status.state = "running"
        self._thread = threading.Thread(target=self._run, daemon=True, name="event-processor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        self.status.state = "stopped"

    def submit(self, session_id: str, candidate: FrameCandidate) -> bool:
        accepted_without_drop = self.queue.put(ProcessingItem(session_id, candidate))
        self.status.queue_depth = len(self.queue)
        self.status.queue_drops = self.queue.drops
        return accepted_without_drop

    def store_one(self, item: ProcessingItem) -> int:
        """フレームを保存してscreen_eventを作成する。OCR・埋め込みは行わない(processed_atはNULLのまま)。

        キャプチャ中はこのメソッドのみを呼び、索引処理(_index_event)はBackgroundIndexerが
        記録停止後にまとめて実行することでCPU負荷を分離する。
        """
        if not self.storage.has_capacity():
            self.status.state = "blocked-storage"
            raise RuntimeError("Storage limit or minimum free-space threshold reached")
        self.status.state = "running"
        stored = self.storage.save_frame(item.candidate.frame, item.session_id, item.candidate.ended_at)
        previous = self.db.fetchone(
            "SELECT id FROM screen_event WHERE session_id=? ORDER BY started_at DESC LIMIT 1",
            (item.session_id,),
        )
        event_id = self.db.execute(
            """
            INSERT INTO screen_event(
                session_id, started_at, ended_at, frame_path, thumbnail_path, phash,
                change_score, width, height, event_kind, previous_event_id, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                item.session_id,
                iso(item.candidate.started_at),
                iso(item.candidate.ended_at),
                str(stored.frame_path),
                str(stored.thumbnail_path),
                perceptual_hash(item.candidate.frame),
                item.candidate.change_score,
                stored.width,
                stored.height,
                item.candidate.event_kind,
                previous["id"] if previous else None,
                iso(datetime.now(UTC)),
            ),
        )
        self.db.execute(
            """INSERT INTO evidence(event_id, kind, path, mime_type, sha256, metadata_json, created_at)
               VALUES(?, 'frame', ?, 'image/webp', ?, '{}', ?)""",
            (event_id, str(stored.frame_path), stored.sha256, iso(datetime.now(UTC))),
        )
        return event_id

    def process_one(self, item: ProcessingItem) -> int:
        """保存 + 索引を同期的に行う(テスト・後方互換用)。通常の記録中はstore_oneのみが使われる。"""
        event_id = self.store_one(item)
        try:
            self._index_event(event_id, item.candidate.frame)
        except Exception as exc:
            LOGGER.exception("Indexing failed for event %s", event_id)
            self.status.last_error = str(exc)
            self.status.failures += 1
        return event_id

    def reprocess_one(self, event_id: int) -> int:
        """Retry OCR and embedding generation for an already stored event."""
        row = self.db.fetchone("SELECT frame_path FROM screen_event WHERE id=?", (event_id,))
        if not row:
            raise LookupError("Event not found")
        try:
            with self.storage.open_image(row["frame_path"]) as image:
                frame = to_bgr(image)
            self._index_event(event_id, frame)
        except Exception as exc:
            LOGGER.exception("Reindexing failed for event %s", event_id)
            self.status.last_error = str(exc)
            self.status.failures += 1
            raise
        self.status.processed += 1
        self.status.last_processed_at = iso(datetime.now(UTC))
        return event_id

    def _index_event(self, event_id: int, frame: np.ndarray) -> None:
        # OCRバックエンドは同時呼び出しを保証しないため、録画処理と手動再処理を直列化する。
        with self._index_lock:
            ocr_result = normalize_ocr_result(self.ocr.recognize(frame))
            vector = self.embeddings.encode_document(ocr_result.text) if ocr_result.text else None
        metadata = json.dumps(
            {
                "provider": self.ocr.name,
                "lines": [
                    {"text": line.text, "confidence": line.confidence, "polygon": line.polygon}
                    for line in ocr_result.lines
                ],
            },
            ensure_ascii=False,
        )
        processed_at = iso(datetime.now(UTC))
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE screen_event SET
                    ocr_text=?, ocr_confidence=?, embedding=?, embedding_dim=?, processed_at=?
                WHERE id=?
                """,
                (
                    ocr_result.text,
                    ocr_result.confidence,
                    vector.astype(np.float32).tobytes() if vector is not None else None,
                    int(vector.size) if vector is not None else None,
                    processed_at,
                    event_id,
                ),
            )
            connection.execute(
                """INSERT INTO evidence(event_id, kind, text, mime_type, metadata_json, created_at)
                   VALUES(?, 'ocr', ?, 'text/plain', ?, ?)""",
                (event_id, ocr_result.text, metadata, processed_at),
            )

    def _run(self) -> None:
        while not self._stop.is_set() or len(self.queue):
            item = self.queue.get(timeout=0.5)
            if item is None:
                continue
            try:
                self.store_one(item)
                self.status.processed += 1
                self.status.last_processed_at = iso(datetime.now(UTC))
            except Exception as exc:
                LOGGER.exception("Event storage failed")
                self.status.failures += 1
                self.status.last_error = str(exc)
            self.status.queue_depth = len(self.queue)
            self.status.queue_drops = self.queue.drops


@dataclass(slots=True)
class IndexerStatus:
    state: str = "idle"  # idle | indexing | paused-capture
    pending_count: int = 0
    indexed_total: int = 0
    failures: int = 0
    last_error: str | None = None
    last_indexed_at: str | None = None


class BackgroundIndexer:
    """未索引(processed_at IS NULL)のイベントを古い順にOCR・埋め込み(索引)処理する。

    CPUでのOCRは1枚あたり数十秒かかる重い処理のため、キャプチャ中に同時実行すると
    CPUを占有しプレビュー・手動キャプチャの応答が遅れる。既定では記録中は待機し、
    停止している間だけ処理する。ただしOCRが別プロセスワーカー(GPU)で動いている
    構成ではメインプロセスの負荷が小さいため、allow_during_captureで記録中の
    処理継続を許可できる(準リアルタイムに検索へ反映される)。
    """

    def __init__(
        self,
        db: Database,
        processor: EventProcessor,
        is_capture_active: Callable[[], bool],
        allow_during_capture: Callable[[], bool] | None = None,
        poll_interval: float = 2.0,
        idle_interval: float = 10.0,
        max_attempts: int = 3,
    ):
        self.db = db
        self.processor = processor
        self.is_capture_active = is_capture_active
        # OCRが別プロセスワーカー(GPU)で動いている等、メインプロセスのCPUを
        # 奪わない構成のときだけ記録中の索引を許可するための判定フック
        self.allow_during_capture = allow_during_capture or (lambda: False)
        self.poll_interval = poll_interval
        self.idle_interval = idle_interval
        self.max_attempts = max_attempts
        self.status = IndexerStatus()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._attempts: dict[int, int] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="background-indexer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)

    def notify(self) -> None:
        """記録停止時などに呼び出し、待機中のポーリング間隔を待たずに再確認させる。"""
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.is_capture_active() and not self.allow_during_capture():
                self.status.state = "paused-capture"
                self._refresh_pending_count()
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            row = self._next_pending()
            if row is None:
                self.status.state = "idle"
                self.status.pending_count = 0
                self._wake.wait(self.idle_interval)
                self._wake.clear()
                continue
            self.status.state = "indexing"
            # 1枚の処理には数十秒かかることがあるため、処理前に残数を更新しておく
            # (起動直後のバックログ処理で「残り0枚」と表示されるのを防ぐ)
            self._refresh_pending_count()
            self._process_row(row)

    def _next_pending(self):
        return self.db.fetchone(
            "SELECT id, frame_path FROM screen_event WHERE processed_at IS NULL ORDER BY started_at LIMIT 1"
        )

    def _refresh_pending_count(self) -> None:
        row = self.db.fetchone("SELECT COUNT(*) AS count FROM screen_event WHERE processed_at IS NULL")
        self.status.pending_count = int(row["count"]) if row else 0

    def _process_row(self, row) -> None:
        event_id = int(row["id"])
        try:
            with self.processor.storage.open_image(row["frame_path"]) as image:
                frame = to_bgr(image)
            self.processor._index_event(event_id, frame)
            self.status.indexed_total += 1
            self.status.last_indexed_at = iso(datetime.now(UTC))
            self._attempts.pop(event_id, None)
        except Exception as exc:
            LOGGER.exception("Background indexing failed for event %s", event_id)
            self.status.failures += 1
            self.status.last_error = str(exc)
            attempts = self._attempts.get(event_id, 0) + 1
            self._attempts[event_id] = attempts
            if attempts >= self.max_attempts:
                # 無限リトライを防ぐため、失敗のまま処理済みとしてマークして次へ進む
                self.db.execute(
                    "UPDATE screen_event SET processed_at=? WHERE id=?",
                    (iso(datetime.now(UTC)), event_id),
                )
                self._attempts.pop(event_id, None)
        finally:
            self._refresh_pending_count()


class RetentionService:
    def __init__(self, db: Database, storage: Storage):
        self.db = db
        self.storage = storage

    def cleanup(self, now: datetime | None = None) -> int:
        cutoff = iso(self.storage.retention_cutoff(now))
        rows = self.db.fetchall(
            """
            SELECT id, frame_path, thumbnail_path FROM screen_event
            WHERE ended_at < ? AND id NOT IN (SELECT event_id FROM context_pack_item)
            """,
            (cutoff,),
        )
        if not rows:
            return 0
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        self.db.execute(f"DELETE FROM screen_event WHERE id IN ({placeholders})", ids)
        self.storage.remove_paths(
            [path for row in rows for path in (row["frame_path"], row["thumbnail_path"])]
        )
        return len(rows)
