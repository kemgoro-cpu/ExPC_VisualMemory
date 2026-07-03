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
