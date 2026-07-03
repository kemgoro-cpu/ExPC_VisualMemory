from datetime import UTC, datetime, timedelta

import numpy as np

from visual_memory.capture import CaptureManager, ChangeDetector, parse_directshow_devices
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
    output = r'''
[dshow @ x] "ELECOM 2MP Webcam" (video)
[dshow @ x]   Alternative name "@device_pnp_camera"
[dshow @ x] "OBS Virtual Camera" (video)
[dshow @ x]   Alternative name "@device_sw_obs"
[dshow @ x] "Microphone" (audio)
[dshow @ x]   Alternative name "@device_cm_microphone"
'''
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
