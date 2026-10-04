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

import cv2
import numpy as np

from .config import Settings
from .imaging import Region, changed_fraction, region_changed_fraction

LOGGER = logging.getLogger(__name__)

# プレビューを自動で止めるまでの秒数。消し忘れてキャプチャーデバイスを掴み続けると
# 記録も開始できなくなるため、放っておいても必ず解放されるようにしている
PREVIEW_SECONDS = 30.0


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
    captured_at: datetime | None = None
    manual_request: ManualCaptureRequest | None = None


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
        self.candidate_captured_at: datetime | None = None
        self.peak_change = 0.0
        self.dirty = False

    def push(self, frame: np.ndarray, monotonic_at: float, captured_at: datetime) -> list[FrameCandidate]:
        if self.previous_frame is None:
            self.previous_frame = frame.copy()
            self.candidate_frame = frame.copy()
            self.last_emit_at = monotonic_at
            self.last_motion_emit_at = monotonic_at
            self.period_started_at = captured_at
            self.candidate_captured_at = captured_at
            return [FrameCandidate(frame.copy(), captured_at, captured_at, 1.0, "initial", captured_at)]

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
            self.candidate_captured_at = captured_at
            self.last_motion_at = monotonic_at
            self.peak_change = max(self.peak_change, score, watch_score)
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
        # A stable image was sampled at the last significant change, before the
        # stability wait ended. Motion/manual/heartbeat images are sampled now.
        observed_at = (self.candidate_captured_at or captured_at) if kind == "stable" else captured_at
        candidate = FrameCandidate(
            frame=frame.copy(),
            started_at=observed_at,
            ended_at=captured_at,
            change_score=self.peak_change,
            event_kind=kind,
            captured_at=observed_at,
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
        # closeは2か所から呼ばれる(停止要求側と、フレーム読み出しループの後始末)。
        # 先頭のNoneチェックだけでは、両方が同時に通り抜けて片方が後始末済みの
        # プロセスを触ってしまうため、まとめて排他する
        self._close_lock = threading.Lock()

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
        with self._stderr_lock:
            self._stderr_tail.clear()
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
        with self._close_lock:
            process = self.process
            if not process:
                return
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for stream in (process.stdout, process.stderr):
                if stream:
                    stream.close()
            if self._stderr_thread and self._stderr_thread.is_alive():
                self._stderr_thread.join(timeout=1)
            self.process = None
            self._stderr_thread = None


class ManualCaptureRequest:
    """One idempotent request, completed by durable storage rather than capture."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.event = threading.Event()
        self._lock = threading.Lock()
        self.state = "pending"
        self.event_id: int | None = None
        self.error: str | None = None

    def finish(self, *, event_id: int | None = None, error: str | None = None) -> None:
        with self._lock:
            if self.state != "pending":
                return
            self.event_id = event_id
            self.error = error
            self.state = "saved" if event_id is not None and error is None else "failed"
            self.event.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "request_id": self.request_id,
                "state": self.state,
                "event_id": self.event_id,
                "error": self.error,
            }


@dataclass(slots=True)
class CaptureStatus:
    state: str = "stopped"
    source_name: str | None = None
    session_id: str | None = None
    last_frame_at: str | None = None
    last_error: str | None = None
    reconnect_attempts: int = 0
    queue_drops: int = 0
    session_started_at: str | None = None
    candidates_emitted: int = 0
    last_candidate_at: str | None = None
    last_candidate_kind: str | None = None
    # プレビュー(記録せず映像だけ確認する状態)。off / starting / running
    preview_state: str = "off"
    preview_seconds_left: int | None = None


class CaptureManager:
    def __init__(
        self,
        settings: Settings,
        on_candidate: Callable[[str, FrameCandidate], bool],
        on_session_start: Callable[[str, str, str], None],
        on_session_end: Callable[[str, str, str | None], None],
        source_factory: Callable[[str], FrameSource] | None = None,
        lookup_manual_capture: Callable[[str], dict | None] | None = None,
    ):
        self.settings = settings
        self.on_candidate = on_candidate
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        self.source_factory = source_factory or (lambda name: FFmpegFrameSource(settings, name))
        self.lookup_manual_capture = lookup_manual_capture
        self.status = CaptureStatus()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: FrameSource | None = None
        self._lock = threading.Lock()
        # プレビュー用に最新フレームの参照だけ保持する(コピーはエンコード時に行う)
        self._preview_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        # 手動キャプチャは別スレッド(APIハンドラ)からの要求をキャプチャスレッドへ橋渡しする
        self._manual_lock = threading.Lock()
        self._manual_requests: dict[str, ManualCaptureRequest] = {}
        self._pending_manual: list[ManualCaptureRequest] = []
        # プレビューは記録とは別系統のスレッドで動く。デバイスは同時に開けないため排他する
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._preview_source: FrameSource | None = None

    def is_recording(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start_preview(self, source_name: str, seconds: float = PREVIEW_SECONDS) -> CaptureStatus:
        """記録せずに映像だけを表示する。``seconds`` 経過したら自動で止まる。

        セッションを作らず、変化検出も保存も行わない。映像が正しく入っているかを
        確かめるためだけの状態。
        """
        with self._lock:
            if self.is_recording():
                raise RuntimeError("Stop capture before starting a preview")
            if self._preview_thread and self._preview_thread.is_alive():
                return self.status
            self._preview_stop.clear()
            self.status.preview_state = "starting"
            self.status.preview_seconds_left = int(seconds)
            self._preview_thread = threading.Thread(
                target=self._run_preview,
                args=(source_name, seconds),
                daemon=True,
                name="capture-preview",
            )
            self._preview_thread.start()
            return self.status

    def stop_preview(self) -> None:
        """プレビューを止め、キャプチャーデバイスが解放されるまで待つ。"""
        self._preview_stop.set()
        source = self._preview_source
        if source:
            source.close()
        thread = self._preview_thread
        if thread and thread.is_alive():
            # 記録開始がこの直後に続くことがあるため、デバイスの解放を待ってから戻る
            thread.join(timeout=8)
        self.status.preview_state = "off"
        self.status.preview_seconds_left = None

    def _run_preview(self, source_name: str, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        try:
            self._preview_source = self.source_factory(source_name)
            self.status.preview_state = "running"
            for frame in self._preview_source.frames(self._preview_stop):
                with self._preview_lock:
                    self._latest_frame = frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.status.preview_seconds_left = max(0, int(remaining))
        except Exception as exc:
            # プレビューは補助機能なので、失敗しても記録状態には影響させない
            self.status.last_error = str(exc)
            LOGGER.warning("Preview failed: %s", exc)
        finally:
            if self._preview_source:
                self._preview_source.close()
                self._preview_source = None
            with self._preview_lock:
                self._latest_frame = None
            self.status.preview_state = "off"
            self.status.preview_seconds_left = None

    def start(self, source_name: str) -> CaptureStatus:
        # プレビュー中でも記録を開始できるようにする(映像を確認してそのまま記録、が自然なため)。
        # デバイスは同時に開けないので、先にプレビューを止めて解放を待つ
        if self._preview_thread and self._preview_thread.is_alive():
            self.stop_preview()
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

    def latest_frame_jpeg(self, max_width: int = 960, quality: int = 80) -> bytes | None:
        """直近のフレームを縮小JPEGへエンコードして返す(要求時にのみエンコードするため負荷は小さい)。"""
        with self._preview_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        height, width = frame.shape[:2]
        if width > max_width:
            scale = max_width / width
            frame = cv2.resize(frame, (max_width, round(height * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return encoded.tobytes()

    def manual_capture_status(self, request_id: str) -> dict | None:
        with self._manual_lock:
            request = self._manual_requests.get(request_id)
        if request is not None:
            return request.snapshot()
        return self.lookup_manual_capture(request_id) if self.lookup_manual_capture else None

    def request_manual_capture(self, timeout: float = 3.0, *, request_id: str | None = None) -> bool | dict:
        """Wait for storage; a timeout leaves a queryable pending request.

        Explicit request IDs return structured results and coalesce retries.
        The legacy no-ID form returns True only after actual storage succeeds.
        """
        structured = request_id is not None
        request_id = request_id or str(uuid.uuid4())
        known = self.manual_capture_status(request_id)
        if known is not None and known["state"] != "pending":
            return known if structured else known["state"] == "saved"
        with self._manual_lock:
            request = self._manual_requests.get(request_id)
            if request is None:
                if self.status.state not in {"running", "reconnecting", "starting"} or self._stop.is_set():
                    raise RuntimeError("Capture is not running")
                request = ManualCaptureRequest(request_id)
                self._manual_requests[request_id] = request
                self._pending_manual.append(request)
        request.event.wait(timeout)
        result = request.snapshot()
        return result if structured else result["state"] == "saved"

    def _run(self, source_name: str) -> None:
        session_id = str(uuid.uuid4())
        self.status.session_id = session_id
        self.status.session_started_at = iso(utcnow())
        self.status.candidates_emitted = 0
        self.status.last_candidate_at = None
        self.status.last_candidate_kind = None
        try:
            self.on_session_start(session_id, source_name, iso(utcnow()))
        except Exception as exc:
            self.status.last_error = str(exc)
            self.status.state = "failed"
            self._release_pending_manual_request()
            LOGGER.exception("Unable to create capture session")
            return
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
                        with self._preview_lock:
                            self._latest_frame = frame
                        candidates = detector.push(frame, time.monotonic(), stamp)
                        for request in self._take_manual_requests():
                            candidates.append(
                                FrameCandidate(frame.copy(), stamp, stamp, 0.0, "manual", stamp, request)
                            )
                        for position, candidate in enumerate(candidates):
                            try:
                                accepted = self.on_candidate(session_id, candidate)
                            except Exception as exc:
                                for remaining in candidates[position:]:
                                    if remaining.manual_request:
                                        remaining.manual_request.finish(error=str(exc))
                                raise
                            if not accepted:
                                self.status.queue_drops += 1
                            self.status.candidates_emitted += 1
                            self.status.last_candidate_at = iso(stamp)
                            self.status.last_candidate_kind = candidate.event_kind
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
            with self._preview_lock:
                self._latest_frame = None
            self._release_pending_manual_request()
            self.on_session_end(session_id, iso(utcnow()), terminal_error)
            self.status.state = "stopped"

    def _take_manual_requests(self) -> list[ManualCaptureRequest]:
        with self._manual_lock:
            requests = self._pending_manual
            self._pending_manual = []
            return requests

    def _release_pending_manual_request(self) -> None:
        for request in self._take_manual_requests():
            request.finish(error="Capture stopped before a frame was submitted")
