import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from visual_memory.capture import CaptureManager, ChangeDetector, FFmpegFrameSource, parse_directshow_devices
from visual_memory.config import Settings
from visual_memory.imaging import Region


def test_change_detector_emits_initial_stable_and_heartbeat():
    detector = ChangeDetector(stable_seconds=1.5, heartbeat_seconds=60, motion_checkpoint_seconds=5)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    black = np.zeros((120, 160, 3), dtype=np.uint8)
    white = np.full_like(black, 255)

    assert [item.event_kind for item in detector.push(black, 0.0, base)] == ["initial"]
    assert detector.push(black, 30.0, base + timedelta(seconds=30)) == []
    assert [item.event_kind for item in detector.push(black, 60.0, base + timedelta(seconds=60))] == [
        "heartbeat"
    ]
    assert detector.push(white, 61.0, base + timedelta(seconds=61)) == []
    assert detector.push(white, 62.0, base + timedelta(seconds=62)) == []
    stable = detector.push(white, 62.6, base + timedelta(seconds=62.6))
    assert [item.event_kind for item in stable] == ["stable"]


def test_continuous_motion_creates_checkpoints():
    detector = ChangeDetector(stable_seconds=1.5, heartbeat_seconds=60, motion_checkpoint_seconds=5)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    detector.push(np.zeros((80, 80, 3), dtype=np.uint8), 0, base)
    emitted = []
    for second in range(1, 12):
        frame = np.full((80, 80, 3), 255 if second % 2 else 0, dtype=np.uint8)
        emitted += detector.push(frame, float(second), base + timedelta(seconds=second))
    assert [item.event_kind for item in emitted] == ["motion", "motion"]


def test_directshow_device_parser_ignores_audio_and_alternative_names():
    output = """
[dshow @ x] "USB3.0 Capture" (video)
[dshow @ x]   Alternative name "@device_pnp_foo"
[dshow @ x] DirectShow audio devices
[dshow @ x] "USB Audio" (audio)
"""
    # Some FFmpeg builds place the video heading before the listed entries.
    output = "[dshow @ x] DirectShow video devices\n" + output
    assert parse_directshow_devices(output) == ["USB3.0 Capture"]


def test_directshow_device_parser_accepts_typed_entries_without_section_headings():
    output = r"""
[dshow @ x] "ELECOM 2MP Webcam" (video)
[dshow @ x]   Alternative name "@device_pnp_camera"
[dshow @ x] "OBS Virtual Camera" (video)
[dshow @ x]   Alternative name "@device_sw_obs"
[dshow @ x] "Microphone" (audio)
[dshow @ x]   Alternative name "@device_cm_microphone"
"""
    assert parse_directshow_devices(output) == ["ELECOM 2MP Webcam", "OBS Virtual Camera"]


def test_ignore_region_suppresses_cursor_or_clock_noise():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    detector = ChangeDetector(ignore_regions=[Region(0.0, 0.0, 0.25, 0.25)])
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[:20, :20] = 255

    detector.push(before, 0.0, base)
    assert detector.push(after, 1.0, base + timedelta(seconds=1)) == []


def test_watch_region_detects_small_but_important_cell_change():
    base = datetime(2026, 1, 1, tzinfo=UTC)
    detector = ChangeDetector(
        change_threshold=0.007,
        watch_change_threshold=0.003,
        watch_regions=[Region(0.4, 0.4, 0.1, 0.1)],
    )
    before = np.zeros((100, 100, 3), dtype=np.uint8)
    after = before.copy()
    after[44:46, 44:46] = 255

    detector.push(before, 0.0, base)
    assert detector.push(after, 1.0, base + timedelta(seconds=1)) == []
    stable = detector.push(after, 2.6, base + timedelta(seconds=2.6))
    assert [item.event_kind for item in stable] == ["stable"]


def test_ffmpeg_stderr_drain_prevents_pipe_deadlock(tmp_path):
    # stdoutを書く前に大量のstderrを出力する疑似FFmpegプロセス。
    # stderrを読み捨てずに放置するとOSのパイプバッファが満杯になり
    # 子プロセスがブロックし、stdout待ちの親側も一緒にハングしてしまう
    settings = Settings(data_dir=tmp_path / "data", capture_width=2, capture_height=2)
    source = FFmpegFrameSource(settings, "dummy")
    frame_bytes = settings.capture_width * settings.capture_height * 3
    script = (
        "import sys\n"
        "sys.stderr.write('W' * 200000)\n"
        "sys.stderr.flush()\n"
        f"sys.stdout.buffer.write(bytes([1]) * {frame_bytes})\n"
        "sys.stdout.flush()\n"
    )
    source.command = lambda: [sys.executable, "-c", script]
    stop_event = threading.Event()
    result: list[np.ndarray] = []

    def consume():
        result.extend(source.frames(stop_event))

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "stderrが未読のままだとFFmpegがブロックしハングする"
    assert len(result) == 1
    source.close()


def test_capture_reconnect_clears_transient_session_error(tmp_path):
    import threading

    settings = Settings(
        data_dir=tmp_path / "data",
        capture_width=8,
        capture_height=8,
        reconnect_seconds=0.01,
    )
    attempts = 0
    reconnected = threading.Event()
    ended = []

    class Source:
        def __init__(self, fail):
            self.fail = fail

        def frames(self, stop_event):
            if self.fail:
                raise RuntimeError("temporary disconnect")
            yield np.zeros((8, 8, 3), dtype=np.uint8)
            reconnected.set()
            stop_event.wait(1)

        def close(self):
            return None

    def source_factory(_name):
        nonlocal attempts
        attempts += 1
        return Source(fail=attempts == 1)

    manager = CaptureManager(
        settings,
        on_candidate=lambda *_: True,
        on_session_start=lambda *_: None,
        on_session_end=lambda *args: ended.append(args),
        source_factory=source_factory,
    )
    manager.start("capture")
    assert reconnected.wait(2)
    manager.stop()

    assert attempts >= 2
    assert ended[0][2] is None
    assert manager.status.last_error is None


def test_latest_frame_jpeg_is_none_before_any_frame(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    manager = CaptureManager(
        settings,
        on_candidate=lambda *_: True,
        on_session_start=lambda *_: None,
        on_session_end=lambda *_: None,
    )
    assert manager.latest_frame_jpeg() is None


def test_manual_capture_requires_running_state(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    manager = CaptureManager(
        settings,
        on_candidate=lambda *_: True,
        on_session_start=lambda *_: None,
        on_session_end=lambda *_: None,
    )
    with pytest.raises(RuntimeError):
        manager.request_manual_capture(timeout=0.1)


def test_manual_capture_emits_candidate_and_updates_progress(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", capture_width=8, capture_height=8)
    candidates = []
    got_first_frame = threading.Event()

    class Source:
        def frames(self, stop_event):
            frame = np.zeros((8, 8, 3), dtype=np.uint8)
            while not stop_event.is_set():
                yield frame
                got_first_frame.set()
                stop_event.wait(0.05)

        def close(self):
            return None

    def on_candidate(_session_id, candidate):
        candidates.append(candidate)
        if candidate.manual_request:
            # This capture-only fixture stands in for a successful storage commit.
            candidate.manual_request.finish(event_id=42)
        return True

    manager = CaptureManager(
        settings,
        on_candidate=on_candidate,
        on_session_start=lambda *_: None,
        on_session_end=lambda *_: None,
        source_factory=lambda _name: Source(),
    )
    manager.start("capture")
    assert got_first_frame.wait(2)

    frame_jpeg = manager.latest_frame_jpeg()
    assert frame_jpeg is not None
    assert frame_jpeg[:2] == b"\xff\xd8"  # JPEGマジックバイト

    fulfilled = manager.request_manual_capture(timeout=2)
    manager.stop()

    assert fulfilled is True
    assert [item.event_kind for item in candidates].count("manual") == 1
    assert manager.status.candidates_emitted >= 2  # 初回自動候補 + 手動候補
    assert manager.status.last_candidate_kind is not None
    assert manager.status.session_started_at is not None


def test_manual_capture_reports_failure_when_capture_stops_while_waiting(tmp_path):
    # 手動キャプチャを待っている間に記録が停止した場合、
    # 「保存成功」ではなくFalse(失敗)が返ることを確認する
    settings = Settings(data_dir=tmp_path / "data", capture_width=8, capture_height=8)
    got_first_frame = threading.Event()

    class Source:
        def frames(self, stop_event):
            # 最初の1フレームだけ流し、その後は停止まで新しいフレームを出さない。
            # 手動キャプチャ要求は次フレーム処理まで実行されないため、待機状態を再現できる
            yield np.zeros((8, 8, 3), dtype=np.uint8)
            got_first_frame.set()
            stop_event.wait(5)

        def close(self):
            return None

    manager = CaptureManager(
        settings,
        on_candidate=lambda *_: True,
        on_session_start=lambda *_: None,
        on_session_end=lambda *_: None,
        source_factory=lambda _name: Source(),
    )
    manager.start("capture")
    assert got_first_frame.wait(2)

    results: list[bool] = []
    waiter = threading.Thread(
        target=lambda: results.append(manager.request_manual_capture(timeout=5)), daemon=True
    )
    waiter.start()
    time.sleep(0.2)  # 要求が登録されるのを待つ
    manager.stop()
    waiter.join(timeout=2)

    assert results == [False]


# --- プレビュー(記録せず映像だけ確認する状態) ---


class _PreviewSource:
    """フレームを流し続けるだけのダミー映像ソース。"""

    def __init__(self, first_frame: threading.Event | None = None):
        self.first_frame = first_frame
        self.closed = False

    def frames(self, stop_event):
        while not stop_event.is_set():
            if self.first_frame is not None:
                self.first_frame.set()
            yield np.zeros((8, 8, 3), dtype=np.uint8)
            time.sleep(0.01)

    def close(self):
        self.closed = True


def _manager(tmp_path, source, **hooks):
    settings = Settings(data_dir=tmp_path / "data", capture_width=8, capture_height=8)
    return CaptureManager(
        settings,
        on_candidate=hooks.get("on_candidate", lambda *_: True),
        on_session_start=hooks.get("on_session_start", lambda *_: None),
        on_session_end=hooks.get("on_session_end", lambda *_: None),
        source_factory=lambda _name: source,
    )


def test_preview_shows_frames_without_recording_anything(tmp_path):
    # プレビューは映像を出すだけで、セッションもイベントも作らない
    started = threading.Event()
    candidates: list = []
    sessions: list = []
    source = _PreviewSource(started)
    manager = _manager(
        tmp_path,
        source,
        on_candidate=lambda *args: candidates.append(args) or True,
        on_session_start=lambda *args: sessions.append(args),
    )

    manager.start_preview("capture", seconds=5)
    try:
        assert started.wait(2)
        assert manager.latest_frame_jpeg() is not None
        assert manager.status.preview_state == "running"
    finally:
        manager.stop_preview()

    assert candidates == []
    assert sessions == []
    assert manager.status.preview_state == "off"
    assert source.closed


def test_preview_stops_by_itself_after_the_given_time(tmp_path):
    # 消し忘れてもキャプチャーデバイスを掴み続けないこと
    source = _PreviewSource()
    manager = _manager(tmp_path, source)

    manager.start_preview("capture", seconds=0.3)
    deadline = time.time() + 5
    while manager.status.preview_state != "off" and time.time() < deadline:
        time.sleep(0.05)

    assert manager.status.preview_state == "off"
    assert source.closed


def test_preview_is_refused_while_recording(tmp_path):
    started = threading.Event()
    manager = _manager(tmp_path, _PreviewSource(started))
    manager.start("capture")
    try:
        assert started.wait(2)
        with pytest.raises(RuntimeError):
            manager.start_preview("capture")
    finally:
        manager.stop()


def test_starting_a_recording_takes_over_from_preview(tmp_path):
    # 「映像を確認 → そのまま記録開始」が押し直しなしで通ること。
    # デバイスは同時に開けないため、記録開始の前にプレビューが解放されている必要がある
    preview_source = _PreviewSource(threading.Event())
    record_source = _PreviewSource(threading.Event())
    sources = [preview_source, record_source]
    settings = Settings(data_dir=tmp_path / "data", capture_width=8, capture_height=8)
    sessions: list = []
    manager = CaptureManager(
        settings,
        on_candidate=lambda *_: True,
        on_session_start=lambda *args: sessions.append(args),
        on_session_end=lambda *_: None,
        source_factory=lambda _name: sources.pop(0),
    )

    manager.start_preview("capture", seconds=30)
    assert preview_source.first_frame.wait(2)
    manager.start("capture")
    try:
        assert record_source.first_frame.wait(2)
        assert manager.status.preview_state == "off"
        assert preview_source.closed  # プレビュー側のデバイスが解放済み
        assert len(sessions) == 1
    finally:
        manager.stop()


def test_frame_source_close_is_safe_from_two_threads(tmp_path):
    """closeは停止要求側と読み出しループの後始末の2か所から同時に呼ばれうる。

    先頭のNoneチェックだけでは両方が通り抜け、片方が後始末したプロセスを
    もう片方が触って AttributeError になる(実機のプレビュー→記録の切り替えで発生した)。
    """
    settings = Settings(data_dir=tmp_path / "data")
    source = FFmpegFrameSource(settings, "dummy")

    class SlowProcess:
        """terminate/waitに時間がかかるプロセスの代役。競合の窓を広げる。"""

        stdout = None
        stderr = None

        def poll(self):
            return None

        def terminate(self):
            time.sleep(0.2)

        def wait(self, timeout=None):
            return 0

    source.process = SlowProcess()
    errors: list[BaseException] = []

    def close():
        try:
            source.close()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert source.process is None


def test_new_stable_screen_uses_its_own_observation_time():
    detector = ChangeDetector()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    before = np.zeros((80, 80, 3), dtype=np.uint8)
    after = np.full_like(before, 255)
    detector.push(before, 0, base)
    detector.push(before, 30, base + timedelta(seconds=30))
    detector.push(after, 31, base + timedelta(seconds=31))
    event = detector.push(after, 32.6, base + timedelta(seconds=32.6))[0]
    assert event.started_at == base + timedelta(seconds=31)
    assert event.captured_at == base + timedelta(seconds=31)
    assert event.ended_at == base + timedelta(seconds=32.6)


def test_motion_checkpoints_use_distinct_sample_times():
    detector = ChangeDetector()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(10)
    events = []
    for second in range(17):
        frame = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        events.extend(detector.push(frame, second, base + timedelta(seconds=second)))
    assert [event.started_at for event in events] == [
        base + timedelta(seconds=second) for second in (0, 6, 11, 16)
    ]
    assert all(event.started_at == event.captured_at == event.ended_at for event in events)


def test_manual_timeout_can_complete_later_without_duplicate_request(tmp_path):
    candidates = []
    ready = threading.Event()
    source = _PreviewSource(ready)
    manager = _manager(tmp_path, source, on_candidate=lambda _, item: candidates.append(item) or True)
    manager.start("capture")
    try:
        assert ready.wait(2)
        first = manager.request_manual_capture(timeout=0.1, request_id="same-request")
        second = manager.request_manual_capture(timeout=0.1, request_id="same-request")
        assert first["state"] == second["state"] == "pending"
        manual = [candidate for candidate in candidates if candidate.event_kind == "manual"]
        assert len(manual) == 1
        manual[0].manual_request.finish(event_id=123)
        assert manager.manual_capture_status("same-request") == {
            "request_id": "same-request",
            "state": "saved",
            "event_id": 123,
            "error": None,
        }
        assert manager.request_manual_capture(request_id="same-request")["event_id"] == 123
        assert len([candidate for candidate in candidates if candidate.event_kind == "manual"]) == 1
    finally:
        manager.stop()


def test_saved_manual_request_can_be_looked_up_after_restart(tmp_path):
    saved = {"request_id": "saved", "state": "saved", "event_id": 7, "error": None}
    manager = CaptureManager(
        Settings(data_dir=tmp_path),
        lambda *_: True,
        lambda *_: None,
        lambda *_: None,
        lookup_manual_capture=lambda request_id: saved if request_id == "saved" else None,
    )
    assert manager.manual_capture_status("saved") == saved
    assert manager.request_manual_capture(request_id="saved") == saved


def test_session_start_failure_releases_waiting_manual_request(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def fail_start(*args):
        entered.set()
        release.wait(2)
        raise RuntimeError("fixture session failure")

    manager = CaptureManager(Settings(data_dir=tmp_path), lambda *_: True, fail_start, lambda *_: None)
    manager.start("capture")
    try:
        assert entered.wait(2)
        assert manager.request_manual_capture(timeout=0, request_id="starting")["state"] == "pending"
        release.set()
        manager._thread.join(timeout=2)
        assert manager.status.state == "failed"
        assert manager.manual_capture_status("starting")["state"] == "failed"
    finally:
        release.set()
        manager.stop()
