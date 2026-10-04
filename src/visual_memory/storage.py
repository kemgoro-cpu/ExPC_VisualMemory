from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .db import Database
from .imaging import sha256_file, to_pil

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StoredFrame:
    frame_path: Path
    thumbnail_path: Path
    sha256: str
    width: int
    height: int
    frame_bytes: int = 0
    thumbnail_bytes: int = 0


class Storage:
    def __init__(self, settings: Settings, db: Database | None = None):
        self.settings = settings
        self.db = db
        self.settings.ensure_directories()
        self._usage_cache_path = self.settings.data_dir / ".storage-usage.json"
        self._usage_lock = threading.Lock()
        self._reconcile_lock = threading.RLock()
        self._reconcile_thread: threading.Thread | None = None
        self._usage_bytes = 0
        self._document_bytes = 0
        self._document_signature = None
        self._usage_state = "pending"
        self._last_reconciled_at: str | None = None
        self._load_usage_cache()

    @property
    def usage_state(self) -> str:
        with self._usage_lock:
            return self._usage_state

    @property
    def last_reconciled_at(self) -> str | None:
        with self._usage_lock:
            return self._last_reconciled_at

    def _load_usage_cache(self) -> None:
        if not self._usage_cache_path.exists():
            return
        try:
            payload = json.loads(self._usage_cache_path.read_text(encoding="utf-8"))
            self._usage_bytes = max(0, int(payload["usage_bytes"]))
            self._document_bytes = max(0, int(payload.get("document_bytes", 0)))
            self._last_reconciled_at = payload.get("reconciled_at")
            self._usage_state = "stale"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._usage_bytes = 0
            self._usage_state = "pending"

    def _write_usage_cache(self) -> None:
        payload = {
            "schema_version": 2,
            "usage_bytes": self._usage_bytes,
            "document_bytes": self._document_bytes,
            "reconciled_at": self._last_reconciled_at,
        }
        temporary = self._usage_cache_path.with_suffix(".json.tmp")
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(self._usage_cache_path)
        except OSError:
            # This is a reconstructible cache. Its failure must not turn a completed
            # frame write into an unregistered orphan or conceal successful storage.
            LOGGER.warning("Unable to persist storage usage cache", exc_info=True)

    def start_reconciliation(self) -> None:
        if self._reconcile_thread and self._reconcile_thread.is_alive():
            return
        with self._usage_lock:
            self._usage_state = "scanning"
        self._reconcile_thread = threading.Thread(
            target=self.reconcile_usage,
            daemon=True,
            name="storage-usage-reconciliation",
        )
        self._reconcile_thread.start()

    def reconcile_usage(self) -> int:
        with self._reconcile_lock:
            total = 0
            documents = 0
            sizes: dict[str, int] = {}
            for root, _, names in os.walk(self.settings.data_dir):
                for name in names:
                    path = Path(root) / name
                    if path == self._usage_cache_path or path.name == ".storage-usage.json.tmp":
                        continue
                    try:
                        size = path.stat().st_size
                        total += size
                        sizes[str(path)] = size
                        if path.is_relative_to(self.settings.packs_dir):
                            documents += size
                    except OSError:
                        continue
            with self._usage_lock:
                self._usage_bytes = total
                self._document_bytes = documents
                self._document_signature = None
                self._usage_state = "ready"
                self._last_reconciled_at = datetime.now(UTC).isoformat()
                self._write_usage_cache()
            if self.db:
                # One migration/backfill during the existing disk reconciliation, not a
                # filesystem scan on every status poll.
                rows = self.db.fetchall("SELECT id,frame_path,thumbnail_path FROM screen_event")
                self.db.executemany(
                    "UPDATE screen_event SET frame_bytes=?,thumbnail_bytes=? WHERE id=?",
                    [
                        (sizes.get(row["frame_path"], 0), sizes.get(row["thumbnail_path"], 0), row["id"])
                        for row in rows
                    ],
                )
            return total

    def invalidate_document_usage(self) -> None:
        with self._reconcile_lock:
            self._document_signature = None

    def document_usage_bytes(self, signature: tuple) -> int:
        with self._reconcile_lock:
            if signature != self._document_signature:
                total = 0
                for root, _, names in os.walk(self.settings.packs_dir):
                    for name in names:
                        try:
                            total += (Path(root) / name).stat().st_size
                        except OSError:
                            continue
                delta = total - self._document_bytes
                self._document_bytes = total
                self._document_signature = signature
                if delta:
                    self._adjust_usage(delta)
            return self._document_bytes

    def _adjust_usage(self, delta: int) -> None:
        with self._usage_lock:
            self._usage_bytes = max(0, self._usage_bytes + delta)
            self._write_usage_cache()

    def save_frame(self, frame: np.ndarray, session_id: str, stamp: datetime) -> StoredFrame:
        day = stamp.astimezone(UTC).strftime("%Y-%m-%d")
        prefix = f"{stamp.astimezone(UTC).strftime('%H%M%S-%f')}-{session_id[:8]}"
        filename = f"{prefix}-{uuid.uuid4().hex[:12]}.webp"
        frame_dir = self.settings.frames_dir / day
        thumb_dir = self.settings.thumbnails_dir / day
        frame_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / filename
        thumbnail_path = thumb_dir / filename

        image = to_pil(frame)
        with self._reconcile_lock:
            try:
                image.save(
                    frame_path, "WEBP", quality=self.settings.webp_quality, method=self.settings.webp_method
                )
                thumb = image.copy()
                thumb.thumbnail((self.settings.thumbnail_width, self.settings.thumbnail_width))
                thumb.save(thumbnail_path, "WEBP", quality=82, method=4)
                frame_bytes = frame_path.stat().st_size
                thumbnail_bytes = thumbnail_path.stat().st_size
                digest = sha256_file(frame_path)
            except Exception:
                frame_path.unlink(missing_ok=True)
                thumbnail_path.unlink(missing_ok=True)
                raise
            self._adjust_usage(frame_bytes + thumbnail_bytes)
        return StoredFrame(
            frame_path=frame_path,
            thumbnail_path=thumbnail_path,
            sha256=digest,
            width=image.width,
            height=image.height,
            frame_bytes=frame_bytes,
            thumbnail_bytes=thumbnail_bytes,
        )

    def usage_bytes(self) -> int:
        with self._usage_lock:
            return self._usage_bytes

    def has_capacity(self) -> bool:
        disk = shutil.disk_usage(self.settings.data_dir)
        return (
            disk.free >= self.settings.minimum_free_bytes
            and self.usage_bytes() < self.settings.max_storage_bytes
        )

    def remove_paths(self, paths: list[str | Path]) -> None:
        root = self.settings.data_dir.resolve()
        removed_bytes = 0
        with self._reconcile_lock:
            for value in paths:
                path = Path(value)
                try:
                    resolved = path.resolve()
                    resolved.relative_to(root)
                except (OSError, ValueError):
                    continue
                try:
                    removed_bytes += resolved.stat().st_size
                    resolved.unlink(missing_ok=True)
                except OSError:
                    continue
            if removed_bytes:
                self._adjust_usage(-removed_bytes)

    def retention_cutoff(self, now: datetime | None = None) -> datetime:
        return (now or datetime.now(UTC)) - timedelta(days=self.settings.retention_days)

    def open_image(self, path: str | Path) -> Image.Image:
        return Image.open(path)
