from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import asdict
from pathlib import Path
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
from .processing import EventProcessor, RetentionService
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
        self._maintenance_stop = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        self._security_lock = threading.Lock()
        self._security_status = {"status": "checking", "detail": "BitLocker check is running"}
        self._security_thread: threading.Thread | None = None

    def start(self) -> None:
        self.storage.start_reconciliation()
        for provider in (self.ocr, self.embeddings):
            start = getattr(provider, "start", None)
            if start:
                start()
        self._start_security_check()
        self.processor.start()
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
            self._start_security_check()

    def _start_security_check(self) -> None:
        if self._security_thread and self._security_thread.is_alive():
            return
        self._security_thread = threading.Thread(
            target=self._refresh_security_status,
            daemon=True,
            name="bitlocker-status",
        )
        self._security_thread.start()

    def _refresh_security_status(self) -> None:
        result = bitlocker_status(self.settings.data_dir)
        with self._security_lock:
            self._security_status = result

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

    def list_devices(self) -> list[str]:
        return list_directshow_devices(self.settings.ffmpeg_path)

    def status(self) -> dict[str, Any]:
        disk = self.settings.disk_usage()
        return {
            "capture": asdict(self.capture.status),
            "processor": asdict(self.processor.status),
            "ocr": {
                "name": self.ocr.name,
                "available": self.ocr.available,
                "reason": self.ocr.reason,
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
            "security": {"bitlocker": self.bitlocker_status()},
        }

    def bitlocker_status(self) -> dict[str, str]:
        with self._security_lock:
            return dict(self._security_status)

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


def bitlocker_status(path: Path) -> dict[str, str]:
    if os.name != "nt":
        return {"status": "not-applicable", "detail": "BitLocker check is Windows-only"}
    drive = path.resolve().drive
    # driveを文字列展開せず、-Command以降の引数を$argsとして渡すことで
    # コマンド文字列組み立てによる注入リスクを避ける
    command = [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-BitLockerVolume -MountPoint $args[0] | "
        "Select-Object ProtectionStatus,VolumeStatus | ConvertTo-Json -Compress",
        drive,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        if result.returncode != 0 or not result.stdout.strip():
            return {"status": "unknown", "detail": result.stderr.strip() or "Unable to query BitLocker"}
        data = json.loads(result.stdout)
        protected = str(data.get("ProtectionStatus", "")).lower() in {"on", "1"}
        return {"status": "protected" if protected else "warning", "detail": json.dumps(data)}
    except Exception as exc:
        return {"status": "unknown", "detail": str(exc)}
