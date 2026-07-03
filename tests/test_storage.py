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
