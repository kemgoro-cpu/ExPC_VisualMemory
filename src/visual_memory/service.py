from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

from .ai import (
    AsyncEmbeddingProvider,
    AsyncOcrProvider,
    build_embedding_provider,
    build_ocr_provider,
)
from .capture import CaptureManager, list_directshow_devices
from .config import Settings
from .db import Database
from .packs import ContextPackService
from .processing import BackgroundIndexer, EventProcessor, RetentionService
from .search import SearchEngine
from .storage import Storage


class VisualMemoryService:
    def __init__(
        self,
        settings: Settings,
        *,
        ocr=None,
        embeddings=None,
        source_factory=None,
    ):
        self.settings = settings
        self.db = Database(settings.database_path)
        self.db.initialize()
        self.storage = Storage(settings)
        self.ocr = ocr or AsyncOcrProvider(
            lambda: build_ocr_provider(
                settings.ocr_provider,
                settings.ocr_detection_model_dir,
                settings.ocr_recognition_model_dir,
                settings.ocr_device,
                settings.ocr_worker_python,
            )
        )
        self.embeddings = embeddings or AsyncEmbeddingProvider(
            lambda: build_embedding_provider(settings.embedding_model)
        )
        self.processor = EventProcessor(self.db, self.storage, self.ocr, self.embeddings)
        self.search = SearchEngine(self.db, self.embeddings)
        self.packs = ContextPackService(self.db, settings)
        self.retention = RetentionService(self.db, self.storage)
        self.capture = CaptureManager(
            settings,
            on_candidate=self.processor.submit,
            on_session_start=self._session_start,
            on_session_end=self._session_end,
            source_factory=source_factory,
        )
        # CPUでのOCR・埋め込みは1枚あたり数十秒かかりCPUを占有するため、既定では記録中は行わず
        # 記録停止後にBackgroundIndexerがまとめて処理する(プレビュー・手動キャプチャの応答性を守るため)。
        # OCRが別プロセスワーカー(GPU)で動いている場合はメインプロセスの負荷が小さいので、
        # 記録中も索引を継続して準リアルタイムに検索へ反映する
        self.indexer = BackgroundIndexer(
            self.db,
            self.processor,
            is_capture_active=lambda: self.capture.status.state not in {"stopped", "failed"},
            allow_during_capture=lambda: self.ocr.name.startswith("worker:"),
        )
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None

    def start(self) -> None:
        self.storage.start_reconciliation()
        for provider in (self.ocr, self.embeddings):
            start = getattr(provider, "start", None)
            if start:
                start()
        self.processor.start()
        self.indexer.start()
        self.packs.expire_due()
        self.retention.cleanup()
        if not self._maintenance_thread or not self._maintenance_thread.is_alive():
            self._maintenance_stop.clear()
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop, daemon=True, name="retention-maintenance"
            )
            self._maintenance_thread.start()

    def stop(self) -> None:
        self.capture.stop()
        self.processor.stop()
        self.indexer.stop()
        close_ocr = getattr(self.ocr, "close", None)
        if close_ocr:
            close_ocr()
        close_embeddings = getattr(self.embeddings, "close", None)
        if close_embeddings:
            close_embeddings()
        self._maintenance_stop.set()
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=3)

    def _maintenance_loop(self) -> None:
        while not self._maintenance_stop.wait(3600):
            self.packs.expire_due()
            self.retention.cleanup()
            self.storage.reconcile_usage()

    def _session_start(self, session_id: str, source_name: str, started_at: str) -> None:
        self.db.execute(
            "INSERT INTO capture_session(id,source_name,started_at,status) VALUES(?,?,?,'running')",
            (session_id, source_name, started_at),
        )

    def _session_end(self, session_id: str, ended_at: str, error: str | None) -> None:
        self.db.execute(
            "UPDATE capture_session SET ended_at=?,status=?,error=? WHERE id=?",
            (ended_at, "failed" if error else "stopped", error, session_id),
        )
        self.indexer.notify()

    def list_devices(self) -> list[str]:
        return list_directshow_devices(self.settings.ffmpeg_path)

    def status(self) -> dict[str, Any]:
        disk = self.settings.disk_usage()
        return {
            "capture": asdict(self.capture.status),
            "processor": asdict(self.processor.status),
            "indexer": asdict(self.indexer.status),
            "ocr": {
                "name": self.ocr.name,
                "available": self.ocr.available,
                "reason": self.ocr.reason,
                "fallback_reason": getattr(self.ocr, "fallback_reason", None),
                "state": getattr(
                    self.ocr,
                    "state",
                    "ready" if self.ocr.available else "disabled",
                ),
            },
            "embeddings": {
                "name": self.embeddings.name,
                "available": self.embeddings.available,
                "dimension": self.embeddings.dimension,
                "reason": self.embeddings.reason,
                "state": getattr(
                    self.embeddings,
                    "state",
                    "ready" if self.embeddings.available else "disabled",
                ),
            },
            "storage": {
                "data_dir": str(self.settings.data_dir),
                "used_bytes": self.storage.usage_bytes(),
                "limit_bytes": self.settings.max_storage_bytes,
                "disk_free_bytes": disk.free,
                "minimum_free_bytes": self.settings.minimum_free_bytes,
                "state": self.storage.usage_state,
                "last_reconciled_at": self.storage.last_reconciled_at,
            },
        }

    def capture_readiness(self) -> tuple[bool, str | None]:
        if self.storage.usage_state in {"pending", "scanning"}:
            return False, "Storage usage is still being checked"
        loading = [
            label
            for label, provider in (("OCR", self.ocr), ("embedding", self.embeddings))
            if getattr(provider, "state", None) == "loading"
        ]
        if loading:
            return False, f"AI models are still loading: {', '.join(loading)}"
        return True, None

    def delete_events(self, start: str | None = None, end: str | None = None) -> int:
        if not start or not end:
            raise ValueError("A bounded start and end time are required")
        conditions = ["id NOT IN (SELECT event_id FROM context_pack_item)"]
        params: list[str] = []
        if start:
            conditions.append("ended_at >= ?")
            params.append(start)
        if end:
            conditions.append("started_at <= ?")
            params.append(end)
        rows = self.db.fetchall(
            f"SELECT id,frame_path,thumbnail_path FROM screen_event WHERE {' AND '.join(conditions)}", params
        )
        if not rows:
            return 0
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        self.db.execute(f"DELETE FROM screen_event WHERE id IN ({placeholders})", ids)
        self.storage.remove_paths(
            [value for row in rows for value in (row["frame_path"], row["thumbnail_path"])]
        )
        return len(rows)

    def reindex_event(self, event_id: int) -> dict[str, Any]:
        if not self.ocr.available:
            raise RuntimeError(self.ocr.reason or "OCR is not available")
        self.processor.reprocess_one(event_id)
        self.search.invalidate_embedding_cache()
        result = self.search.event_with_neighbors(event_id)
        if not result:
            raise LookupError("Event not found")
        return result
