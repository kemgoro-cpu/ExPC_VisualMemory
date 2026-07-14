from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Settings
from .imaging import sha256_file, to_pil


@dataclass(frozen=True, slots=True)
class StoredFrame:
    frame_path: Path
    thumbnail_path: Path
    sha256: str
    width: int
    height: int


class Storage:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_directories()
        self._usage_cache_path = self.settings.data_dir / ".storage-usage.json"
        self._usage_lock = threading.Lock()
        self._reconcile_lock = threading.RLock()
        self._reconcile_thread: threading.Thread | None = None
        self._usage_bytes = 0
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
            self._last_reconciled_at = payload.get("reconciled_at")
            self._usage_state = "stale"
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            self._usage_bytes = 0
            self._usage_state = "pending"

    def _write_usage_cache(self) -> None:
        payload = {
            "schema_version": 1,
            "usage_bytes": self._usage_bytes,
            "reconciled_at": self._last_reconciled_at,
        }
        temporary = self._usage_cache_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self._usage_cache_path)

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
            for root, _, names in os.walk(self.settings.data_dir):
                for name in names:
                    path = Path(root) / name
                    if path == self._usage_cache_path or path.name == ".storage-usage.json.tmp":
                        continue
                    try:
                        total += path.stat().st_size
                    except OSError:
                        continue
            with self._usage_lock:
                self._usage_bytes = total
                self._usage_state = "ready"
                self._last_reconciled_at = datetime.now(UTC).isoformat()
                self._write_usage_cache()
            return total

    def _adjust_usage(self, delta: int) -> None:
        with self._usage_lock:
            self._usage_bytes = max(0, self._usage_bytes + delta)
            self._write_usage_cache()

    def save_frame(self, frame: np.ndarray, session_id: str, stamp: datetime) -> StoredFrame:
        day = stamp.astimezone(UTC).strftime("%Y-%m-%d")
        filename = f"{stamp.astimezone(UTC).strftime('%H%M%S-%f')}-{session_id[:8]}.webp"
        frame_dir = self.settings.frames_dir / day
        thumb_dir = self.settings.thumbnails_dir / day
        frame_dir.mkdir(parents=True, exist_ok=True)
        thumb_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / filename
        thumbnail_path = thumb_dir / filename

        image = to_pil(frame)
        with self._reconcile_lock:
            image.save(
                frame_path, "WEBP", quality=self.settings.webp_quality, method=self.settings.webp_method
            )
            thumb = image.copy()
            thumb.thumbnail((self.settings.thumbnail_width, self.settings.thumbnail_width))
            thumb.save(thumbnail_path, "WEBP", quality=82, method=4)
            self._adjust_usage(frame_path.stat().st_size + thumbnail_path.stat().st_size)
        return StoredFrame(
            frame_path=frame_path,
            thumbnail_path=thumbnail_path,
            sha256=sha256_file(frame_path),
            width=image.width,
            height=image.height,
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
