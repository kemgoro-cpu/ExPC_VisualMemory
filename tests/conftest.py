from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from visual_memory.ai import OcrLine, OcrResult
from visual_memory.capture import FrameCandidate
from visual_memory.config import Settings
from visual_memory.service import VisualMemoryService


class QueueOcr:
    name = "fake-ocr"
    available = True
    reason = None

    def __init__(self, texts: list[str] | None = None):
        self.texts = list(texts or [])

    def recognize(self, frame):
        text = self.texts.pop(0) if self.texts else ""
        return OcrResult(
            text=text, confidence=0.99 if text else None, lines=[OcrLine(text, 0.99, [])] if text else []
        )


class HashEmbedding:
    name = "fake-embedding"
    available = True
    reason = None
    dimension = 32

    def _encode(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in text.lower().replace("query:", "").replace("passage:", "").split():
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % self.dimension
            vector[index] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else None

    def encode_document(self, text: str):
        return self._encode(text)

    def encode_query(self, text: str):
        return self._encode(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(data_dir=tmp_path / "data", capture_width=64, capture_height=36)
    value.ensure_directories()
    return value


@pytest.fixture
def service(settings: Settings) -> VisualMemoryService:
    return VisualMemoryService(settings, ocr=QueueOcr(), embeddings=HashEmbedding())


def add_event(service: VisualMemoryService, text: str, seconds: int = 0, color: int = 30) -> int:
    from datetime import UTC, datetime, timedelta

    if not service.db.fetchone("SELECT id FROM capture_session WHERE id='session'"):
        service.db.execute(
            """INSERT INTO capture_session(id,source_name,started_at,status)
               VALUES('session','test',?,'stopped')""",
            (datetime(2026, 1, 1, tzinfo=UTC).isoformat(),),
        )
    service.ocr.texts.append(text)
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    candidate = FrameCandidate(
        frame=np.full((36, 64, 3), color, dtype=np.uint8),
        started_at=stamp,
        ended_at=stamp,
        change_score=0.5,
        event_kind="stable",
    )
    return service.processor.process_one(
        type("Item", (), {"session_id": "session", "candidate": candidate})()
    )
