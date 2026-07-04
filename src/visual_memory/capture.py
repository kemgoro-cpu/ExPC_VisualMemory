from __future__ import annotations

import logging
import re
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings
from .imaging import Region, changed_fraction, region_changed_fraction

LOGGER = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


@dataclass(slots=True)
class FrameCandidate:
    frame: np.ndarray
    started_at: datetime
    ended_at: datetime
    change_score: float
    event_kind: str


class ChangeDetector:
    def __init__(
        self,
        stable_seconds: float = 1.5,
        heartbeat_seconds: float = 60.0,
        motion_checkpoint_seconds: float = 5.0,
        change_threshold: float = 0.007,
        watch_change_threshold: float = 0.003,
        pixel_delta: int = 18,
        ignore_regions: list[Region] | None = None,
        watch_regions: list[Region] | None = None,
    ):
        self.stable_seconds = stable_seconds
        self.heartbeat_seconds = heartbeat_seconds
        self.motion_checkpoint_seconds = motion_checkpoint_seconds
        self.change_threshold = change_threshold
        self.watch_change_threshold = watch_change_threshold
        self.pixel_delta = pixel_delta
        self.ignore_regions = list(ignore_regions or [])
        self.watch_regions = list(watch_regions or [])
        self.previous_frame: np.ndarray | None = None
        self.candidate_frame: np.ndarray | None = None
        self.last_motion_at: float | None = None
        self.last_emit_at: float | None = None
        self.last_motion_emit_at: float | None = None
        self.period_started_at: datetime | None = None
        self.peak_change = 0.0
        self.dirty = False

    def push(self, frame: np.ndarray, monotonic_at: float, captured_at: datetime) -> list[FrameCandidate]:
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            self.candidate_frame = frame.copy()
            self.last_emit_at = monotonic_at
            self.last_motion_emit_at = monotonic_at
            self.period_started_at = captured_at
            return [FrameCandidate(frame.copy(), captured_at, captured_at, 1.0, "initial")]

        score = changed_fraction(self.previous_frame, frame, self.pixel_delta, self.ignore_regions)
        watch_score = max(
            (
                region_changed_fraction(self.previous_frame, frame, self.pixel_delta, region)
                for region in self.watch_regions
            ),
            default=0.0,
        )
        self.previous_frame = frame.copy()
        results: list[FrameCandidate] = []
        if score >= self.change_threshold or watch_score >= self.watch_change_threshold:
            starting_motion = not self.dirty
            self.dirty = True
            self.candidate_frame = frame.copy()
            self.last_motion_at = monotonic_at
            self.peak_change = max(self.peak_change, score, watch_score)
            if self.period_started_at is None:
                self.period_started_at = captured_at
            if starting_motion:
                self.last_motion_emit_at = monotonic_at
            if (
                self.last_motion_emit_at is None
                or monotonic_at - self.last_motion_emit_at >= self.motion_checkpoint_seconds
            ):
                results.append(self._emit(frame, captured_at, monotonic_at, "motion"))
                self.last_motion_emit_at = monotonic_at
            return results

        if (
            self.dirty
            and self.last_motion_at is not None
            and monotonic_at - self.last_motion_at >= self.stable_seconds
        ):
            stable_frame = self.candidate_frame if self.candidate_frame is not None else frame
            results.append(self._emit(stable_frame, captured_at, monotonic_at, "stable"))
            self.dirty = False
            self.peak_change = 0.0
            self.period_started_at = captured_at
            return results

        if (
            not self.dirty
            and self.last_emit_at is not None
            and monotonic_at - self.last_emit_at >= self.heartbeat_seconds
        ):
            results.append(self._emit(frame, captured_at, monotonic_at, "heartbeat"))
            self.period_started_at = captured_at
        return results

    def _emit(
        self, frame: np.ndarray, captured_at: datetime, monotonic_at: float, kind: str
    ) -> FrameCandidate:
        candidate = FrameCandidate(
            frame=frame.copy(),
            started_at=self.period_started_at or captured_at,
            ended_at=captured_at,
            change_score=self.peak_change,
            event_kind=kind,
        )
        self.last_emit_at = monotonic_at
        return candidate


class FrameSource(Protocol):
    def frames(self, stop_event: threading.Event) -> Iterator[np.ndarray]: ...

    def close(self) -> None: ...


def parse_directshow_devices(stderr: str) -> list[str]:
    devices: list[str] = []
    in_video = False
    for line in stderr.splitlines():
        lowered = line.lower()
        if "alternative name" in lowered:
            continue

        typed_entry = re.search(r'"([^"]+)"\s+\((video|audio)\)\s*$', line, re.IGNORECASE)
        if typed_entry:
            if typed_entry.group(2).lower() == "video" and typed_entry.group(1) not in devices:
                devices.append(typed_entry.group(1))
            continue

        if "directshow video devices" in lowered:
            in_video = True
            continue
        if "directshow audio devices" in lowered:
            in_video = False
            continue
        if not in_video:
            continue
        match = re.search(r'"([^"]+)"', line)
        if match and match.group(1) not in devices:
            devices.append(match.group(1))
    return devices


def list_directshow_devices(ffmpeg_path: str = "ffmpeg") -> list[str]:
    result = subprocess.run(
        [ffmpeg_path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    return parse_directshow_devices(result.stderr)


STDERR_TAIL_BYTES = 4096


class FFmpegFrameSource:
    def __init__(self, settings: Settings, source_name: str, source_file: str | Path | None = None):
        self.settings = settings
        self.source_name = source_name
        self.source_file = Path(source_file) if source_file else None
        self.process: subprocess.Popen[bytes] | None = None
        # stderrを別スレッドで読み続けないと、FFmpegが警告を大量出力した際に
        # OSのパイプバッファが満杯になりFFmpeg側がブロックしてハングするため、
        # 末尾だけをリングバッファ的に保持してエラーメッセージ用に使う
        self._stderr_tail = bytearray()
        self._stderr_lock = threading.Lock()
        self._stderr_thread: threading.Thread | None = None

    def command(self) -> list[str]:
        base = [self.settings.ffmpeg_path, "-hide_banner", "-loglevel", "warning"]
        if self.source_file:
            base += ["-re", "-stream_loop", "-1", "-i", str(self.source_file)]
        else:
            base += [
                "-f",
                "dshow",
                "-rtbufsize",
                "512M",
                "-video_size",
                f"{self.settings.capture_width}x{self.settings.capture_height}",
                "-framerate",
                str(self.settings.capture_fps),
                "-i",
                f"video={self.source_name}",
            ]
        base += [
            "-an",
            "-vf",
            f"fps={self.settings.analysis_fps},scale={self.settings.capture_width}:{self.settings.capture_height}:flags=lanczos",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        return base

    def frames(self, stop_event: threading.Event) -> Iterator[np.ndarray]:
        frame_size = self.settings.capture_width * self.settings.capture_height * 3
        self.process = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self.process.stderr,), daemon=True, name="ffmpeg-stderr"
        )
        self._stderr_thread.start()
        while not stop_event.is_set():
            data = self._read_exact(self.process.stdout, frame_size)
            if len(data) != frame_size:
                break
            yield np.frombuffer(data, dtype=np.uint8).reshape(
                (self.settings.capture_height, self.settings.capture_width, 3)
            )
        return_code = self.process.poll()
        if not stop_event.is_set() and return_code not in (None, 0):
            with self._stderr_lock:
                error = bytes(self._stderr_tail)
            raise RuntimeError(error.decode("utf-8", errors="replace") or f"FFmpeg exited {return_code}")

    def _drain_stderr(self, stream) -> None:
        # 常時読み切ることでOSパイプバッファの詰まりを防ぐ。内容は末尾のみ保持する
        while True:
            chunk = stream.read(4096)
            if not chunk:
                break
            with self._stderr_lock:
                self._stderr_tail.extend(chunk)
                if len(self._stderr_tail) > STDERR_TAIL_BYTES:
                    del self._stderr_tail[: len(self._stderr_tail) - STDERR_TAIL_BYTES]

    @staticmethod
    def _read_exact(stream, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


@dataclass(slots=True)
class CaptureStatus:
    state: str = "stopped"
    source_name: str | None = None
    session_id: str | None = None
    last_frame_at: str | None = None
    last_error: str | None = None
    reconnect_attempts: int = 0
    queue_drops: int = 0


class CaptureManager:
    def __init__(
        self,
        settings: Settings,
        on_candidate: Callable[[str, FrameCandidate], bool],
        on_session_start: Callable[[str, str, str], None],
        on_session_end: Callable[[str, str, str | None], None],
        source_factory: Callable[[str], FrameSource] | None = None,
    ):
        self.settings = settings
        self.on_candidate = on_candidate
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        self.source_factory = source_factory or (lambda name: FFmpegFrameSource(settings, name))
        self.status = CaptureStatus()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: FrameSource | None = None
        self._lock = threading.Lock()

    def start(self, source_name: str) -> CaptureStatus:
        with self._lock:
            if self._thread and self._thread.is_alive():
                raise RuntimeError("Capture is already running")
            self._stop.clear()
            self.status = CaptureStatus(state="starting", source_name=source_name)
            self._thread = threading.Thread(
                target=self._run, args=(source_name,), daemon=True, name="capture"
            )
            self._thread.start()
            return self.status

    def stop(self) -> None:
        self._stop.set()
        if self._source:
            self._source.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=8)
        self.status.state = "stopped"

    def _run(self, source_name: str) -> None:
        session_id = str(uuid.uuid4())
        self.status.session_id = session_id
        self.on_session_start(session_id, source_name, iso(utcnow()))
        detector = ChangeDetector(
            stable_seconds=self.settings.stable_seconds,
            heartbeat_seconds=self.settings.heartbeat_seconds,
            motion_checkpoint_seconds=self.settings.motion_checkpoint_seconds,
            change_threshold=self.settings.change_threshold,
            watch_change_threshold=self.settings.watch_change_threshold,
            pixel_delta=self.settings.pixel_delta,
            ignore_regions=[Region(**item).normalized() for item in self.settings.ignore_regions],
            watch_regions=[Region(**item).normalized() for item in self.settings.watch_regions],
        )
        terminal_error: str | None = None
        try:
            while not self._stop.is_set():
                try:
                    self.status.state = "running" if self.status.reconnect_attempts == 0 else "reconnecting"
                    self._source = self.source_factory(source_name)
                    for frame in self._source.frames(self._stop):
                        stamp = utcnow()
                        terminal_error = None
                        self.status.last_error = None
                        self.status.last_frame_at = iso(stamp)
                        self.status.state = "running"
                        for candidate in detector.push(frame, time.monotonic(), stamp):
                            if not self.on_candidate(session_id, candidate):
                                self.status.queue_drops += 1
                    if not self._stop.is_set():
                        raise RuntimeError("Capture stream ended")
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    terminal_error = str(exc)
                    self.status.last_error = terminal_error
                    self.status.state = "reconnecting"
                    self.status.reconnect_attempts += 1
                    LOGGER.warning("Capture failed; retrying: %s", exc)
                    self._stop.wait(self.settings.reconnect_seconds)
                finally:
                    if self._source:
                        self._source.close()
                        self._source = None
        finally:
            self.on_session_end(session_id, iso(utcnow()), terminal_error)
            self.status.state = "stopped"
