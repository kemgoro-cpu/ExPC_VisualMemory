from conftest import add_event
from fastapi.testclient import TestClient

from visual_memory.api import create_app


def test_api_requires_auth_and_csrf(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        event_id = add_event(service, "searchable")
        assert client.get("/health").status_code == 200
        assert client.get("/api/events").status_code == 401
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        assert response.status_code == 303
        client.cookies.update(response.cookies)
        index = client.get("/")
        assert "/static/app.js?v=" in index.text
        assert "/static/app.css?v=" in index.text
        static = client.get("/static/app.js")
        assert static.headers["cache-control"] == "no-store, max-age=0"
        assert client.get("/api/events?q=searchable").status_code == 200
        assert "security" not in client.get("/api/status").json()
        events_payload = client.get("/api/events?q=searchable").json()
        assert events_payload["sessions"][0]["id"] == "session"
        assert events_payload["sessions"][0]["event_count"] == 1
        assert client.post("/api/packs", json={"title": "x", "event_ids": [1]}).status_code == 403
        allowed = client.post(
            "/api/packs",
            json={"title": "x", "event_ids": [event_id]},
            headers={"X-CSRF-Token": settings.csrf_token},
        )
        assert allowed.status_code == 200
        assert allowed.json()["status"] == "approved"
        assert allowed.json()["deduplicate_overlaps"] == 1

        legacy = client.post(
            "/api/packs",
            json={
                "title": "without deduplication",
                "event_ids": [event_id],
                "deduplicate_overlaps": False,
            },
            headers={"X-CSRF-Token": settings.csrf_token},
        )
        assert legacy.status_code == 200
        assert legacy.json()["deduplicate_overlaps"] == 0
        assert (
            client.put(
                f"/api/packs/{allowed.json()['id']}/items/{event_id}/redactions",
                json={"redactions": []},
                headers={"X-CSRF-Token": settings.csrf_token},
            ).status_code
            == 404
        )


def test_created_pack_is_immediately_available(settings, service):
    event_id = add_event(service, "available")
    pack = service.packs.create("available", [event_id])
    assert pack["status"] == "approved"
    assert service.packs.get_approved(pack["id"])["id"] == pack["id"]


def test_approved_pack_downloads_as_single_html_document(settings, service):
    event_id = add_event(service, "document evidence")
    pack = service.packs.create("document", [event_id])
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        document = client.get(f"/api/packs/{pack['id']}/document?format=html")
        assert document.status_code == 200
        assert document.headers["content-type"].startswith("text/html")
        assert "data:image/webp;base64," in document.text


def test_capture_regions_require_csrf_and_persist(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        payload = {
            "ignore_regions": [{"x": 0, "y": 0, "width": 0.2, "height": 0.1}],
            "watch_regions": [{"x": 0.4, "y": 0.4, "width": 0.2, "height": 0.2}],
        }
        assert client.put("/api/capture/regions", json=payload).status_code == 403
        saved = client.put(
            "/api/capture/regions",
            json=payload,
            headers={"X-CSRF-Token": settings.csrf_token},
        )
        assert saved.status_code == 200
        assert client.get("/api/capture/regions").json() == payload
        assert (settings.data_dir / "config.json").exists()


def test_status_reports_background_indexer_state(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        status = client.get("/api/status").json()
        assert "indexer" in status
        assert set(status["indexer"]) == {
            "state",
            "pending_count",
            "indexed_total",
            "failures",
            "last_error",
            "last_indexed_at",
        }


def test_status_reports_ocr_fallback_reason(settings, service):
    # GPUワーカー失敗時のCPUフォールバック理由(通常はNone)がUIへ届くこと
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        status = client.get("/api/status").json()
        assert "fallback_reason" in status["ocr"]
        assert status["ocr"]["fallback_reason"] is None


def test_capture_preview_and_manual_require_running_capture(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        assert client.get("/api/capture/preview").status_code == 409
        assert (
            client.post(
                "/api/capture/manual", headers={"X-CSRF-Token": settings.csrf_token}
            ).status_code
            == 409
        )


def test_event_history_supports_incremental_paging(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        for index in range(65):
            add_event(service, f"history {index}", seconds=index)
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        first = client.get("/api/events?limit=40&offset=0").json()
        second = client.get(f"/api/events?limit=40&offset={first['next_offset']}").json()

    assert len(first["events"]) == 40
    assert first["has_more"] is True
    assert first["next_offset"] == 40
    assert first["total"] == 65
    assert len(second["events"]) == 25
    assert second["has_more"] is False
    assert {item["id"] for item in first["events"]}.isdisjoint(
        item["id"] for item in second["events"]
    )


def test_query_token_is_only_accepted_on_root(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        assert client.get(f"/api/events?token={settings.auth_token}").status_code == 401
        assert client.get(f"/?token={settings.auth_token}", follow_redirects=False).status_code == 303


def test_delete_events_requires_bounded_range_and_confirmation(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        event_id = add_event(service, "delete me")
        headers = {"X-CSRF-Token": settings.csrf_token}
        assert client.delete("/api/events", headers=headers).status_code == 400
        assert (
            client.delete(
                "/api/events?start=2026-01-01T00:00:00Z&end=2026-01-01T00:01:00Z",
                headers=headers,
            ).status_code
            == 400
        )
        deleted = client.delete(
            "/api/events?start=2026-01-01T00:00:00Z&end=2026-01-01T00:01:00Z&confirm=true",
            headers=headers,
        )

    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": 1}
    assert service.db.fetchone("SELECT id FROM screen_event WHERE id=?", (event_id,)) is None


def test_event_can_be_reindexed_after_initial_ocr_failure(monkeypatch, settings, service):
    original_recognize = service.processor.ocr.recognize

    def fail_once(_frame):
        raise RuntimeError("temporary OCR failure")

    monkeypatch.setattr(service.processor.ocr, "recognize", fail_once)
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        event_id = add_event(service, "discarded after failure")
        assert service.db.fetchone("SELECT processed_at FROM screen_event WHERE id=?", (event_id,))[
            "processed_at"
        ] is None

        service.processor.ocr.texts.clear()
        service.processor.ocr.texts.append("recovered OCR text")
        monkeypatch.setattr(service.processor.ocr, "recognize", original_recognize)
        reindexed = client.post(
            f"/api/events/{event_id}/reindex",
            headers={"X-CSRF-Token": settings.csrf_token},
        )

    assert reindexed.status_code == 200
    assert reindexed.json()["event"]["ocr_text"] == "recovered OCR text"
    assert service.search.search("recovered OCR text")[0]["id"] == event_id


def test_reindex_does_not_erase_ocr_when_provider_is_unavailable(monkeypatch, settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        event_id = add_event(service, "keep existing OCR")
        monkeypatch.setattr(service.ocr, "available", False)
        monkeypatch.setattr(service.ocr, "reason", "OCR model is unavailable")
        response = client.post(
            f"/api/events/{event_id}/reindex",
            headers={"X-CSRF-Token": settings.csrf_token},
        )

    assert response.status_code == 409
    assert service.db.fetchone("SELECT ocr_text FROM screen_event WHERE id=?", (event_id,))[
        "ocr_text"
    ] == "keep existing OCR"


def test_ui_exposes_recovery_and_keyboard_controls(settings, service):
    app = create_app(settings, service)
    with TestClient(app) as client:
        response = client.get(f"/?token={settings.auth_token}", follow_redirects=False)
        client.cookies.update(response.cookies)
        script = client.get("/static/app.js").text

    assert 'tabindex="0" role="checkbox"' in script
    assert "OCR・検索を再処理" in script
    assert "文書を再生成" in script
    assert "statusInterval = null" in script
    assert "墨消し" not in script
    assert "BitLocker" not in script
