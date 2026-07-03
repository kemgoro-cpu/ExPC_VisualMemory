from datetime import UTC, datetime

from conftest import add_event


def test_retention_deletes_old_unreferenced_events_but_preserves_pack_items(service):
    old_id = add_event(service, "old", seconds=0)
    kept_id = add_event(service, "kept", seconds=1)
    service.packs.create("keep", [kept_id])
    future = datetime(2026, 3, 1, tzinfo=UTC)
    deleted = service.retention.cleanup(now=future)
    assert deleted == 1
    assert service.db.fetchone("SELECT id FROM screen_event WHERE id=?", (old_id,)) is None
    assert service.db.fetchone("SELECT id FROM screen_event WHERE id=?", (kept_id,)) is not None
