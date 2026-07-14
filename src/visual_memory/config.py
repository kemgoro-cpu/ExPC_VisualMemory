from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import sys
from contextlib import suppress
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

APP_NAME = "ExternalPCVisualMemory"


def _default_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / APP_NAME


def _default_ffmpeg_path() -> str:
    if getattr(sys, "frozen", False):
        roots = [Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)), Path(sys.executable).parent]
        for root in roots:
            bundled = root / "ffmpeg.exe"
            if bundled.exists():
                return str(bundled)
    return "ffmpeg"


def _bundled_model_path(*parts: str) -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "models"
    candidate = root.joinpath(*parts)
    return candidate if candidate.exists() else None


def _default_embedding_model() -> str:
    bundled = _bundled_model_path("multilingual-e5-small")
    return str(bundled) if bundled else "intfloat/multilingual-e5-small"


def _default_ocr_detection_dir() -> str | None:
    bundled = _bundled_model_path("paddlex", "PP-OCRv6_medium_det")
    return str(bundled) if bundled else None


def _default_ocr_recognition_dir() -> str | None:
    bundled = _bundled_model_path("paddlex", "PP-OCRv6_medium_rec")
    return str(bundled) if bundled else None


@dataclass(slots=True)
class Settings:
    data_dir: Path = _default_data_dir()
    host: str = "127.0.0.1"
    port: int = 8765
    ffmpeg_path: str = _default_ffmpeg_path()
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 30
    analysis_fps: float = 2.0
    stable_seconds: float = 1.5
    heartbeat_seconds: float = 60.0
    motion_checkpoint_seconds: float = 5.0
    change_threshold: float = 0.007
    watch_change_threshold: float = 0.003
    pixel_delta: int = 18
    ignore_regions: list[dict[str, float]] = field(default_factory=list)
    watch_regions: list[dict[str, float]] = field(default_factory=list)
    retention_days: int = 30
    max_storage_bytes: int = 20 * 1024**3
    minimum_free_bytes: int = 2 * 1024**3
    default_approval_hours: int = 24
    max_pack_images: int = 200
    pack_warning_images: int = 50
    webp_quality: int = 90
    # WebPのmethod=4は圧縮率が高い分エンコードが遅い(1080pで実測約350ms)。
    # 記録中は保存が頻繁に発生しCPUを取り合うため、2(実測約130ms)に下げて
    # プレビュー配信やAPI応答に回せるCPU時間を優先する
    webp_method: int = 2
    thumbnail_width: int = 360
    ocr_provider: str = "paddle"
    ocr_device: str = "auto"
    ocr_worker_python: str | None = None
    ocr_detection_model_dir: str | None = field(default_factory=_default_ocr_detection_dir)
    ocr_recognition_model_dir: str | None = field(default_factory=_default_ocr_recognition_dir)
    embedding_model: str = field(default_factory=_default_embedding_model)
    reconnect_seconds: float = 5.0

    @property
    def database_path(self) -> Path:
        return self.data_dir / "visual-memory.sqlite3"

    @property
    def frames_dir(self) -> Path:
        return self.data_dir / "frames"

    @property
    def thumbnails_dir(self) -> Path:
        return self.data_dir / "thumbnails"

    @property
    def packs_dir(self) -> Path:
        return self.data_dir / "context-packs"

    @property
    def auth_token_path(self) -> Path:
        return self.data_dir / ".auth-token"

    @property
    def csrf_token_path(self) -> Path:
        return self.data_dir / ".csrf-token"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.frames_dir, self.thumbnails_dir, self.packs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def secret(self, path: Path) -> str:
        self.ensure_directories()
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
        value = secrets.token_urlsafe(32)
        path.write_text(value, encoding="utf-8")
        with suppress(OSError):
            path.chmod(0o600)
        return value

    @property
    def auth_token(self) -> str:
        return self.secret(self.auth_token_path)

    @property
    def csrf_token(self) -> str:
        return self.secret(self.csrf_token_path)

    def disk_usage(self) -> shutil._ntuple_diskusage:
        self.ensure_directories()
        return shutil.disk_usage(self.data_dir)

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["data_dir"] = str(self.data_dir)
        return result

    def save_region_config(
        self, ignore_regions: list[dict[str, float]], watch_regions: list[dict[str, float]]
    ) -> None:
        config_path = self.data_dir / "config.json"
        current: dict[str, Any] = {}
        if config_path.exists():
            try:
                current = json.loads(config_path.read_text(encoding="utf-8"))
                if not isinstance(current, dict):
                    raise ValueError("config root must be a JSON object")
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                # 破損したconfig.jsonは無視し、デフォルトから作り直す
                LOGGER.warning("config.json is corrupted; recreating with defaults: %s", exc)
                current = {}
        current["ignore_regions"] = ignore_regions
        current["watch_regions"] = watch_regions
        temporary = config_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(config_path)
        self.ignore_regions = ignore_regions
        self.watch_regions = watch_regions


def load_settings(data_dir: str | Path | None = None) -> Settings:
    root = Path(data_dir or os.environ.get("VISUAL_MEMORY_DATA_DIR") or _default_data_dir())
    values: dict[str, Any] = {"data_dir": root}
    config_path = root / "config.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config root must be a JSON object")
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            # 破損したconfig.jsonで起動不能にせず、デフォルト設定にフォールバックする
            LOGGER.warning("config.json is corrupted; falling back to defaults: %s", exc)
            raw = {}
        allowed = {field.name for field in fields(Settings)} - {"data_dir"}
        values.update({key: value for key, value in raw.items() if key in allowed})
    env_map = {
        "VISUAL_MEMORY_HOST": ("host", str),
        "VISUAL_MEMORY_PORT": ("port", int),
        "VISUAL_MEMORY_FFMPEG": ("ffmpeg_path", str),
        "VISUAL_MEMORY_RETENTION_DAYS": ("retention_days", int),
        "VISUAL_MEMORY_OCR_PROVIDER": ("ocr_provider", str),
        "VISUAL_MEMORY_OCR_DEVICE": ("ocr_device", str),
        "VISUAL_MEMORY_OCR_WORKER_PYTHON": ("ocr_worker_python", str),
    }
    for env_name, (key, converter) in env_map.items():
        if env_name in os.environ:
            values[key] = converter(os.environ[env_name])
    settings = Settings(**values)
    settings.ensure_directories()
    return settings
