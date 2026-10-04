from datetime import UTC, datetime

import numpy as np

from visual_memory.storage import Storage


def test_storage_usage_is_cached_and_adjusted_without_rescanning(monkeypatch, settings):
    storage = Storage(settings)
    baseline = storage.reconcile_usage()
    assert storage.usage_state == "ready"

    def unexpected_walk(*_args, **_kwargs):
        raise AssertionError("usage cache should not rescan files")

    monkeypatch.setattr("visual_memory.storage.os.walk", unexpected_walk)
    stored = storage.save_frame(
        np.full((36, 64, 3), 120, dtype=np.uint8),
        "session",
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    after_save = storage.usage_bytes()
    assert after_save > baseline
    assert storage.has_capacity()

    storage.remove_paths([stored.frame_path, stored.thumbnail_path])
    assert storage.usage_bytes() == baseline


def test_storage_usage_cache_survives_restart(settings):
    storage = Storage(settings)
    expected = storage.reconcile_usage()
    restored = Storage(settings)

    assert restored.usage_bytes() == expected
    assert restored.usage_state == "stale"


def test_same_sample_can_be_saved_for_distinct_requests_without_overwrite(settings):
    storage = Storage(settings)
    stamp = datetime(2026, 1, 1, tzinfo=UTC)
    frame = np.full((36, 64, 3), 120, dtype=np.uint8)
    first = storage.save_frame(frame, "session", stamp)
    second = storage.save_frame(frame, "session", stamp)
    assert first.frame_path != second.frame_path
    assert first.thumbnail_path != second.thumbnail_path
    assert first.frame_path.exists() and second.frame_path.exists()
    assert first.frame_bytes == first.frame_path.stat().st_size


def test_failed_thumbnail_write_cleans_partial_frame(settings, monkeypatch):
    import pytest
    from PIL import Image

    storage = Storage(settings)
    original = Image.Image.save

    def fail_thumbnail(image, path, *args, **kwargs):
        if str(settings.thumbnails_dir) in str(path):
            raise OSError("disk full")
        return original(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_thumbnail)
    with pytest.raises(OSError, match="disk full"):
        storage.save_frame(np.zeros((36, 64, 3), dtype=np.uint8), "session", datetime.now(UTC))
    assert not list(settings.frames_dir.rglob("*.webp"))
    assert storage.usage_bytes() == 0


def test_status_storage_categories_do_not_scan_on_every_poll(service, monkeypatch):
    from conftest import add_event

    kept = add_event(service, "keep", seconds=1)
    add_event(service, "normal history", seconds=2)
    service.packs.create("keep", [kept])
    first = service.status()["storage"]
    assert first["document_bytes"] > 0
    assert first["retained_source_bytes"] > 0
    assert first["history_bytes"] > 0
    assert first["retained_event_count"] == 1

    def unexpected_walk(*args, **kwargs):
        raise AssertionError("Repeated status must use cached file sizes")

    monkeypatch.setattr("visual_memory.storage.os.walk", unexpected_walk)
    assert service.status()["storage"]["document_bytes"] == first["document_bytes"]


def test_usage_cache_write_failure_does_not_lose_a_saved_frame(settings, monkeypatch):
    from pathlib import Path

    storage = Storage(settings)
    original = Path.write_text

    def fail_cache(path, *args, **kwargs):
        if path.name == ".storage-usage.json.tmp":
            raise OSError("sidecar write failed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_cache)
    stored = storage.save_frame(np.zeros((36, 64, 3), dtype=np.uint8), "session", datetime.now(UTC))
    assert stored.frame_path.exists() and stored.thumbnail_path.exists()
    assert storage.usage_bytes() == stored.frame_bytes + stored.thumbnail_bytes
