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


def test_revoke_expire_and_document_delete_preserve_sources_until_explicit_delete(service):
    from pathlib import Path

    kept = add_event(service, "long term source")
    pack = service.packs.create("keep", [kept])
    service.packs.revoke(pack["id"])
    assert service.retention.cleanup(datetime(2027, 1, 1, tzinfo=UTC)) == 0
    assert service.packs.export_document(pack["id"], "html").exists()
    service.packs.delete(pack["id"])
    assert service.retention.cleanup(datetime(2027, 1, 1, tzinfo=UTC)) == 0
    row = service.db.fetchone("SELECT * FROM screen_event WHERE id=?", (kept,))
    assert row["retained"] == 1
    assert Path(row["frame_path"]).exists()
    assert service.delete_events("2025-12-31T00:00:00+00:00", "2026-01-02T00:00:00+00:00") == 1
    assert not Path(row["frame_path"]).exists()
