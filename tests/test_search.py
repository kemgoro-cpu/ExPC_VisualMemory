from conftest import add_event
from visual_memory.api import _normalize_time


def test_exact_and_semantic_search(service):
    retry_id = add_event(service, "def retry_request timeout backoff", seconds=1, color=10)
    margin_id = add_event(service, "Q3 gross margin improved profitability", seconds=2, color=20)
    add_event(service, "Windows restart required", seconds=3, color=30)

    exact = service.search.search("retry_request")
    assert exact[0]["id"] == retry_id
    semantic = service.search.search("gross margin profitability")
    assert semantic[0]["id"] == margin_id


def test_time_filter_and_recent_timeline(service):
    first = add_event(service, "first", seconds=1)
    second = add_event(service, "second", seconds=20)
    rows = service.search.search(start="2026-01-01T00:00:10+00:00")
    assert [row["id"] for row in rows] == [second]
    detail = service.search.event_with_neighbors(second)
    assert detail["before"][0]["id"] == first


def test_japanese_substring_search_uses_trigram_with_short_query_fallback(service):
    event_id = add_event(service, "ECU噴射マップを更新しました", seconds=1)

    assert service.search.search("噴射マップ")[0]["id"] == event_id
    assert service.search.search("噴射")[0]["id"] == event_id


def test_normalize_time_z_suffix_matches_db_stored_offset_format(service):
    first = add_event(service, "first", seconds=1)
    second = add_event(service, "second", seconds=20)

    normalized = _normalize_time("2026-01-01T00:00:10Z")
    assert normalized == "2026-01-01T00:00:10+00:00"

    rows = service.search.search(start=normalized)
    assert [row["id"] for row in rows] == [second]
    assert first not in [row["id"] for row in rows]


def test_mixed_query_short_token_is_not_dropped_from_exact_search(service):
    short_only_id = add_event(service, "会議メモを更新しました", seconds=1)
    long_only_id = add_event(service, "report2024 quarterly summary", seconds=2)

    rows = service.search._exact("会議 report2024", None, None, 50)
    ids = {int(row["id"]) for row in rows}

    assert short_only_id in ids
    assert long_only_id in ids


def test_initialize_migrates_unicode61_fts_and_rebuilds_existing_rows(service):
    event_id = add_event(service, "既存の燃料噴射マップ", seconds=1)
    with service.db.connect() as connection:
        connection.executescript(
            """
            DROP TRIGGER screen_event_ai;
            DROP TRIGGER screen_event_ad;
            DROP TRIGGER screen_event_au;
            DROP TABLE screen_event_fts;
            CREATE VIRTUAL TABLE screen_event_fts USING fts5(
                ocr_text, content='screen_event', content_rowid='id', tokenize='unicode61'
            );
            """
        )

    service.db.initialize()

    definition = service.db.fetchone(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='screen_event_fts'"
    )["sql"]
    assert "trigram" in definition.lower()
    assert service.search.search("燃料噴射")[0]["id"] == event_id
