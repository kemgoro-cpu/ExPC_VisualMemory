import json
from datetime import UTC, datetime, timedelta

import pytest
from conftest import add_event

from visual_memory import packs as packs_module


def test_external_program_environment_resets_pyinstaller_dll_directory(
    monkeypatch, tmp_path
):
    bundle_root = (tmp_path / "bundle").resolve()
    bundled_child = bundle_root / "nested"
    system_path = (tmp_path / "system").resolve()
    bundle_root.mkdir()
    bundled_child.mkdir()
    system_path.mkdir()
    dll_directory_calls: list[str | None] = []

    monkeypatch.setattr(packs_module.sys, "platform", "win32")
    monkeypatch.setattr(packs_module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(packs_module.sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setenv(
        "PATH",
        packs_module.os.pathsep.join(
            (str(bundle_root), str(bundled_child), str(system_path))
        ),
    )
    monkeypatch.setattr(
        packs_module, "_set_windows_dll_directory", dll_directory_calls.append
    )

    with packs_module._external_program_environment() as environment:
        assert dll_directory_calls == [None]
        assert environment["PATH"] == packs_module.os.pathsep.join(
            (str(bundle_root), str(bundled_child), str(system_path))
        )

    assert dll_directory_calls == [None, str(bundle_root)]


def test_looks_like_chrome_noise_detects_combined_menu_bar_line():
    assert packs_module._looks_like_chrome_noise(
        "表示(V) 移動(G) 実行(R) ターミナル(T) ヘルプ(H)"
    )


def test_looks_like_chrome_noise_detects_single_known_menu_item():
    assert packs_module._looks_like_chrome_noise("表示(V)")
    assert packs_module._looks_like_chrome_noise("ヘルプ(H)")


def test_looks_like_chrome_noise_does_not_flag_ordinary_text_with_parens():
    assert not packs_module._looks_like_chrome_noise("結果(暫定)を確認する")
    assert not packs_module._looks_like_chrome_noise("四半期の売上高は前年比12.5%増")


def test_important_ocr_excludes_menu_bar_lines():
    text = (
        "月次売上レポート\n"
        "表示(V) 移動(G) 実行(R) ターミナル(T) ヘルプ(H)\n"
        "第1四半期の売上高は前年比12.5%増"
    )
    important = packs_module._important_ocr(text)
    assert "月次売上レポート" in important
    assert "第1四半期の売上高は前年比12.5%増" in important
    assert not any("移動(G)" in line for line in important)


def test_pack_is_immediately_available(service):
    event_id = add_event(service, "visible text", color=255)
    pack = service.packs.create("Review", [event_id], note="Only this frame")
    assert service.packs.get_approved(pack["id"])["id"] == pack["id"]


def test_legacy_redactions_remain_effective_when_rebuilding_existing_data(service):
    event_id = add_event(service, "secret line\npublic line", color=255)
    metadata = {
        "provider": "fake-ocr",
        "lines": [
            {
                "text": "secret line",
                "confidence": 0.99,
                "polygon": [[0, 0], [30, 0], [30, 20], [0, 20]],
            },
            {
                "text": "public line",
                "confidence": 0.99,
                "polygon": [[34, 0], [63, 0], [63, 20], [34, 20]],
            },
        ],
    }
    service.db.execute(
        "UPDATE evidence SET metadata_json=? WHERE event_id=? AND kind='ocr'",
        (json.dumps(metadata), event_id),
    )
    pack = service.packs.create("Scrub OCR", [event_id])
    service.db.execute(
        "UPDATE context_pack_item SET redactions_json=? WHERE pack_id=? AND event_id=?",
        (
            json.dumps([{"x": 0.0, "y": 0.0, "width": 0.5, "height": 1.0}]),
            pack["id"],
            event_id,
        ),
    )
    approved = service.packs.approve(pack["id"])

    assert approved["items"][0]["ocr_text"] == "public line"
    artifact_dir = service.packs.artifact_dir(approved)
    manifest = (artifact_dir / "manifest.json").read_text(encoding="utf-8")
    markdown = (artifact_dir / "context.md").read_text(encoding="utf-8")
    document = (artifact_dir / "context.html").read_text(encoding="utf-8")
    assert "secret line" not in manifest
    assert "secret line" not in markdown
    assert "secret line" not in document
    assert "public line" in manifest
    assert "public line" in document


def test_expired_and_revoked_packs_are_not_listed_for_mcp(service):
    event_id = add_event(service, "approved evidence")
    pack = service.packs.create("Temporary", [event_id])
    past = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    service.db.execute("UPDATE context_pack SET expires_at=? WHERE id=?", (past, pack["id"]))
    assert service.packs.list(include_drafts=False) == []
    with pytest.raises(PermissionError):
        service.packs.get_approved(pack["id"])
    service.packs.revoke(pack["id"])
    with pytest.raises(PermissionError):
        service.packs.get_approved(pack["id"])


def test_reapproving_approved_pack_with_expiry_hours_extends_expiration(service):
    event_id = add_event(service, "extend me")
    pack = service.packs.create("Extend", [event_id])
    original_expires_at = datetime.fromisoformat(pack["expires_at"])

    extended = service.packs.approve(pack["id"], expiry_hours=999)
    new_expires_at = datetime.fromisoformat(extended["expires_at"])

    assert new_expires_at > original_expires_at


def test_reapproval_starts_explicit_expiry_from_now(service):
    event_id = add_event(service, "renew me")
    pack = service.packs.create("Renew", [event_id])
    old_approved = datetime.now(UTC) - timedelta(days=30)
    service.db.execute(
        "UPDATE context_pack SET approved_at=?, expires_at=? WHERE id=?",
        (old_approved.isoformat(), (datetime.now(UTC) + timedelta(days=1)).isoformat(), pack["id"]),
    )

    before = datetime.now(UTC)
    renewed = service.packs.approve(pack["id"], expiry_hours=2)
    after = datetime.now(UTC)
    approved_at = datetime.fromisoformat(renewed["approved_at"])
    expires_at = datetime.fromisoformat(renewed["expires_at"])

    assert before <= approved_at <= after
    assert expires_at - approved_at == timedelta(hours=2)


def test_export_is_one_self_contained_html_file(service):
    event_id = add_event(service, "export me")
    pack = service.packs.create("Export", [event_id])
    target = service.packs.export_document(pack["id"], "html")
    document = target.read_text(encoding="utf-8")
    assert target.name == "context.html"
    assert "data:image/webp;base64," in document
    assert "export me" in document
    assert 'src="images/' not in document


def test_pdf_is_lazily_generated_from_html(monkeypatch, service):
    event_id = add_event(service, "pdf export")
    pack = service.packs.create("PDF", [event_id])

    def fake_render(_html_path, target):
        target.write_bytes(b"%PDF-1.7\n")

    monkeypatch.setattr(service.packs, "_render_pdf", fake_render)
    target = service.packs.export_document(pack["id"], "pdf")
    assert target.read_bytes().startswith(b"%PDF")


def test_context_document_accepts_more_than_twelve_events(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(13)]
    pack = service.packs.create("Thirteen slides", event_ids)
    assert len(pack["items"]) == 13


def test_context_document_preserves_requested_order(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(3)]
    requested = list(reversed(event_ids))
    pack = service.packs.create("Reverse order", requested)
    assert [item["event_id"] for item in pack["items"]] == requested


def test_manifest_v3_records_overlap_setting_and_page_provenance(service):
    event_id = add_event(service, "manifest provenance")
    pack = service.packs.create("Manifest v3", [event_id])
    manifest = json.loads(
        (service.packs.artifact_dir(pack) / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == 3
    assert manifest["deduplicate_overlaps"] is True
    assert manifest["items"][0]["source_event_ids"] == [event_id]
    assert manifest["items"][0]["stitch"]["applied"] is False


def test_automatic_title_prefers_clean_japanese_file_name(service):
    event_ids = [
        add_event(service, "効率化グループの価值可视化.pptx-PowerPoint", seconds=0),
        add_event(service, "効率化グループの価値可視化.pptx-PowerPoint", seconds=1),
    ]
    pack = service.packs.create("", event_ids)
    assert pack["title"] == "効率化グループの価値可視化"


def test_automatic_title_ignores_mid_sentence_extension_mention(service):
    # 「rendererではNode.jsを無効にし…」のような文中の言及はファイル名候補として
    # 採用しない(以前は「rendererではNode」という拡張子除去済みの断片が
    # タイトルになっていた)。ファイル名候補が無いので通常の候補行にフォールバックし、
    # 途中で不自然に切れたタイトルにはならない
    event_id = add_event(
        service,
        "rendererではNode.jsを無効にし、context isolationとsandboxを有効化する。",
    )
    pack = service.packs.create("", [event_id])
    assert pack["title"] != "rendererではNode"


def test_automatic_title_extracts_basename_from_editor_tab(service):
    event_ids = [
        add_event(service, "PLAN.md X", seconds=0),
        add_event(service, "PLAN.md X", seconds=1),
    ]
    pack = service.packs.create("", event_ids)
    assert pack["title"] == "PLAN"


def test_automatic_title_extracts_basename_from_breadcrumb_path(service):
    event_ids = [
        add_event(
            service,
            "C: > Users > kemgo > Documents > Program > DevControlTower > PLAN.md",
            seconds=0,
        ),
        add_event(
            service,
            "C: > Users > kemgo > Documents > Program > DevControlTower > PLAN.md",
            seconds=1,
        ),
    ]
    pack = service.packs.create("", event_ids)
    assert pack["title"] == "PLAN"


def test_html_has_compact_event_pages_ocr_appendix_and_technical_page(service):
    event_id = add_event(service, "重要な見出し\n100万円の効果\n次のアクション")
    pack = service.packs.create("Compact PDF", [event_id])
    document = service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8")
    assert "01 / 01" in document
    assert "重要OCR" in document
    assert "抽出テキスト付録" in document
    assert "DOCUMENT INTEGRITY" in document
    assert "Manifest SHA-256" in document
    assert 'class="technical"' in document


def test_edge_generates_real_pdf_document(service):
    if not service.packs._edge_path():
        pytest.skip("Microsoft Edge is not available")
    event_id = add_event(service, "PDFのOCRテキスト")
    pack = service.packs.create("Real PDF", [event_id])
    target = service.packs.export_document(pack["id"], "pdf")
    assert target.read_bytes().startswith(b"%PDF")
    assert target.stat().st_size > 1000


def test_failed_rebuild_is_unpublished_and_keeps_previous_artifact(monkeypatch, service):
    event_id = add_event(service, "sensitive document")
    pack = service.packs.create("Atomic rebuild", [event_id])
    previous_dir = service.packs.artifact_dir(pack)

    def fail_render(*_args, **_kwargs):
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr(packs_module, "render_document_pages", fail_render)
    with pytest.raises(RuntimeError, match="simulated render failure"):
        service.packs.approve(pack["id"])

    failed = service.packs.get(pack["id"])
    assert failed["status"] == "draft"
    assert "simulated render failure" in failed["build_error"]
    assert previous_dir.exists()
    assert not list(service.settings.packs_dir.glob(f".{pack['id']}-*.tmp"))
    with pytest.raises(PermissionError):
        service.packs.get_approved(pack["id"])
