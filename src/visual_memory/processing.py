from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from .ai import EmbeddingProvider, OcrProvider
from .capture import FrameCandidate, iso
from .db import Database
from .imaging import perceptual_hash
from .storage import Storage

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

    def process_one(self, item: ProcessingItem) -> int:
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
        try:
            ocr_result = self.ocr.recognize(item.candidate.frame)
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
                        iso(datetime.now(UTC)),
                        event_id,
                    ),
                )
                connection.execute(
                    """INSERT INTO evidence(event_id, kind, text, mime_type, metadata_json, created_at)
                       VALUES(?, 'ocr', ?, 'text/plain', ?, ?)""",
                    (event_id, ocr_result.text, metadata, iso(datetime.now(UTC))),
                )
        except Exception as exc:
            LOGGER.exception("Indexing failed for event %s", event_id)
            self.status.last_error = str(exc)
            self.status.failures += 1
        return event_id

    def _run(self) -> None:
        while not self._stop.is_set() or len(self.queue):
            item = self.queue.get(timeout=0.5)
            if item is None:
                continue
            try:
                self.process_one(item)
                self.status.processed += 1
                self.status.last_processed_at = iso(datetime.now(UTC))
            except Exception as exc:
                LOGGER.exception("Event processing failed")
                self.status.failures += 1
                self.status.last_error = str(exc)
            self.status.queue_depth = len(self.queue)
            self.status.queue_drops = self.queue.drops


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
