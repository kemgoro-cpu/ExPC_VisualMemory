import html
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
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


# looks_like_chrome_noise 自体の単体テストは tests/test_textnorm.py にある
# (判定ロジックは索引処理と共有するため visual_memory.textnorm へ移動した)


def _ocr_line(text: str, top: int = 0, height: int = 20, confidence: float = 0.99):
    """重要OCRの選定テスト用に、1行分のOCRデータ(座標つき)を組み立てる。"""
    return {
        "text": text,
        "polygon": [[0, top], [200, top], [200, top + height], [0, top + height]],
        "confidence": confidence,
    }


def test_important_ocr_drops_icon_misreadings_but_keeps_short_headings():
    lines = [
        _ocr_line("乙〇0", top=0),  # アイコンの誤認識
        _ocr_line("@]", top=20),
        _ocr_line("100", top=40),
        _ocr_line("結論", top=60),  # 短いが意味のある見出し
        _ocr_line("第1章", top=80),
        _ocr_line("変速タイミングは車両ごとに算出される", top=100),
    ]
    important = packs_module._important_ocr(lines)
    assert important == ["結論", "第1章", "変速タイミングは車両ごとに算出される"]


def test_important_ocr_drops_lines_the_ocr_was_unsure_about():
    lines = [
        _ocr_line("制造老#号-..扣中约z", top=0, confidence=0.50),
        _ocr_line("必要入力データは製造者が申告する", top=20, confidence=0.98),
    ]
    assert packs_module._important_ocr(lines) == ["必要入力データは製造者が申告する"]


def test_important_ocr_picks_up_a_heading_below_the_body_text():
    """本文より明らかに大きい行は、画面の下のほうにあっても拾われる。"""
    lines = [_ocr_line(f"本文の{index}行目です", top=index * 20) for index in range(10)]
    lines.append(_ocr_line("3.シフトアップの明示ルール", top=200, height=40))

    important = packs_module._important_ocr(lines, limit=8)

    assert "3.シフトアップの明示ルール" in important
    # 点数順ではなく画面の並び順で返す(見出しが先頭に来てしまうと文脈が壊れる)
    assert important[-1] == "3.シフトアップの明示ルール"
    assert important[0] == "本文の0行目です"


def test_important_ocr_accepts_plain_text_for_older_manifests():
    """polygon も confidence も持たない古いmanifestからでも選定できる。"""
    important = packs_module._important_ocr("見出しの行\n本文の行\n@]")
    assert important == ["見出しの行", "本文の行"]


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


@pytest.mark.parametrize("cancel", ["revoke", "delete"])
def test_build_claim_spans_service_instances_and_cancellation(monkeypatch, service, cancel):
    event_id = add_event(service, "Document being rebuilt")
    pack = service.packs.create("Concurrent rebuild", [event_id])
    second = packs_module.ContextPackService(service.db, service.settings)
    entered, release = threading.Event(), threading.Event()
    original = packs_module.render_document_pages

    def blocked(sources, deduplicate):
        entered.set()
        assert release.wait(10)
        return original(sources, deduplicate)

    monkeypatch.setattr(packs_module, "render_document_pages", blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(service.packs.approve, pack["id"])
        try:
            assert entered.wait(10)
            with pytest.raises(packs_module.PackError, match="already being rebuilt"):
                second.approve(pack["id"])
            getattr(second, cancel)(pack["id"])
        finally:
            release.set()
        with pytest.raises(packs_module.PackError, match="cancelled or superseded"):
            pending.result(timeout=10)
    assert second.list(include_drafts=False) == []
    if cancel == "revoke":
        assert second.get(pack["id"])["status"] == "revoked"
    else:
        with pytest.raises(packs_module.PackError, match="not found"):
            second.get(pack["id"])


@pytest.mark.parametrize("old_fails", [False, True])
def test_expired_build_lease_cannot_overwrite_newer_success(monkeypatch, service, old_fails):
    event_id = add_event(service, "Newest document must survive")
    pack = service.packs.create("Lease recovery", [event_id])
    second = packs_module.ContextPackService(service.db, service.settings)
    entered, release = threading.Event(), threading.Event()
    original = packs_module.render_document_pages

    def delayed_old(sources, deduplicate):
        if threading.current_thread().name.startswith("old-build"):
            entered.set()
            assert release.wait(10)
            if old_fails:
                raise RuntimeError("late old failure")
        return original(sources, deduplicate)

    monkeypatch.setattr(packs_module, "render_document_pages", delayed_old)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="old-build") as pool:
        pending = pool.submit(service.packs.approve, pack["id"])
        try:
            assert entered.wait(10)
            service.db.execute("UPDATE context_pack SET build_started_at=? WHERE id=?",
                               ((datetime.now(UTC) - timedelta(hours=1)).isoformat(), pack["id"]))
            newest = second.configure_and_rebuild(
                pack["id"], deduplicate_overlaps=False, include_appendix=True,
            )
        finally:
            release.set()
        with pytest.raises(packs_module.PackError, match="cancelled or superseded"):
            pending.result(timeout=10)
    current = second.get_approved(pack["id"])
    assert current["artifact_path"] == newest["artifact_path"]
    assert current["build_error"] is None
    assert current["include_appendix"] == 1
    assert second.export_document(pack["id"], "html").exists()


@pytest.mark.parametrize("status", ["approved", "revoked", "expired"])
def test_reconfigure_preserves_order_and_sharing_deadline(service, status):
    first = add_event(service, "First selected original", seconds=1)
    second = add_event(service, "Second selected original", seconds=2)
    pack = service.packs.create("Keep sources", [second, first])
    if status == "revoked":
        pack = service.packs.revoke(pack["id"])
    elif status == "expired":
        service.db.execute("UPDATE context_pack SET expires_at=? WHERE id=?",
                           ((datetime.now(UTC) - timedelta(hours=1)).isoformat(), pack["id"]))
        pack = service.packs.get(pack["id"])
    updated = service.packs.configure_and_rebuild(
        pack["id"], deduplicate_overlaps=False, include_appendix=True,
    )
    assert updated["status"] == status
    assert updated["expires_at"] == pack["expires_at"]
    assert updated["approved_at"] == pack["approved_at"]
    assert [item["event_id"] for item in updated["items"]] == [second, first]
    assert updated["include_appendix"] == 1
    assert updated["deduplicate_overlaps"] == 0
    if status != "approved":
        with pytest.raises(PermissionError):
            service.packs.get_approved(pack["id"])


def test_creation_failure_exposes_same_document_for_retry(monkeypatch, service):
    event_id = add_event(service, "Retry this document")
    original = packs_module.render_document_pages

    def fail(*_args):
        raise RuntimeError("render unavailable")

    monkeypatch.setattr(packs_module, "render_document_pages", fail)
    with pytest.raises(packs_module.PackError, match="render unavailable") as caught:
        service.packs.create("Failed once", [event_id])
    pack_id = caught.value.pack_id
    assert pack_id
    assert service.packs.get(pack_id)["status"] == "draft"
    monkeypatch.setattr(packs_module, "render_document_pages", original)
    assert service.packs.approve(pack_id)["status"] == "approved"
    assert len(service.packs.list()) == 1


def test_explicit_document_deletion_retains_originals_and_other_documents(service):
    first = add_event(service, "Shared original", seconds=1)
    second = add_event(service, "Unique original", seconds=2)
    removing = service.packs.create("Remove this document", [first, second])
    keeping = service.packs.create("Keep this document", [first])
    info = service.packs.deletion_info(removing["id"])
    assert info["source_count"] == 2
    assert info["shared_source_count"] == 1
    assert info["document_bytes"] > 0
    result = service.packs.delete(removing["id"])
    assert result["deleted"]
    assert result["removed_document_bytes"] == info["document_bytes"]
    assert result["remaining_document_bytes"] == 0
    assert service.packs.export_document(keeping["id"], "html").exists()
    rows = service.db.fetchall("SELECT retained,frame_path FROM screen_event ORDER BY id")
    assert len(rows) == 2
    assert all(row["retained"] == 1 for row in rows)
    assert all(packs_module.Path(row["frame_path"]).exists() for row in rows)
    assert service.db.fetchone("SELECT COUNT(*) AS n FROM context_pack_item WHERE pack_id=?",
                              (removing["id"],))["n"] == 0


def test_mcp_export_rechecks_revoke_after_slow_pdf_render(monkeypatch, service):
    event_id = add_event(service, "Stop during PDF preparation")
    pack = service.packs.create("Slow PDF", [event_id])

    def render(_html_path, target):
        service.packs.revoke(pack["id"])
        target.write_bytes(b"%PDF-1.7\n")

    monkeypatch.setattr(service.packs, "_render_pdf", render)
    with pytest.raises(PermissionError):
        service.packs.get_approved_document(pack["id"], "pdf")
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


def test_pack_records_the_page_count_it_actually_built(service):
    # 選んだ画面の枚数(item_count)と、重複をまとめたあとのページ数は別物。
    # ページ数を持たないと、利用者からは重複除去が効いているか分からない
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(3)]
    pack = service.packs.create("Page count", event_ids)
    manifest = json.loads(
        (service.packs.artifact_dir(pack) / "manifest.json").read_text(encoding="utf-8")
    )

    assert pack["page_count"] == len(manifest["items"])
    assert pack["page_count"] <= len(event_ids)
    listed = next(item for item in service.packs.list() if item["id"] == pack["id"])
    assert listed["page_count"] == pack["page_count"]
    assert listed["item_count"] == len(event_ids)


def test_pack_detail_exposes_built_pages_not_source_screens(service):
    # 画面に並べるのは「出来上がったページ」。元画面の件数(items)とは別に持つ
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(3)]
    pack = service.packs.create("Pages", event_ids)
    manifest = json.loads(
        (service.packs.artifact_dir(pack) / "manifest.json").read_text(encoding="utf-8")
    )

    assert len(pack["pages"]) == len(manifest["items"]) == pack["page_count"]
    assert [page["position"] for page in pack["pages"]] == list(range(len(pack["pages"])))
    covered = {
        event_id for page in pack["pages"] for event_id in page["source_event_ids"]
    }
    assert covered == set(event_ids)


def test_page_image_is_served_from_inside_the_document_folder(service):
    event_id = add_event(service, "page image")
    pack = service.packs.create("Page image", [event_id])

    path = service.packs.page_image_path(pack["id"], 0)
    assert path.exists()
    assert path.is_relative_to(service.packs.artifact_dir(pack).resolve())
    with pytest.raises(packs_module.PackError):
        service.packs.page_image_path(pack["id"], 999)


def test_removing_pages_drops_their_screens_in_one_rebuild(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(4)]
    pack = service.packs.create("Removable", event_ids)
    removed = {
        event_id
        for page in pack["pages"][:2]
        for event_id in page["source_event_ids"]
    }

    updated = service.packs.remove_pages(pack["id"], [0, 1])

    assert updated["page_count"] == pack["page_count"] - 2
    # 選んだページの元画面だけが消え、選ばなかったものは残る
    assert {item["event_id"] for item in updated["items"]} == set(event_ids) - removed
    # 外したのは文書からだけで、画面履歴には残る
    assert service.db.fetchone("SELECT id FROM screen_event WHERE id=?", (min(removed),))


def test_removing_no_page_is_rejected(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(2)]
    pack = service.packs.create("Nothing selected", event_ids)

    with pytest.raises(packs_module.PackError):
        service.packs.remove_pages(pack["id"], [])
    assert len(service.packs.get(pack["id"])["items"]) == len(event_ids)


def test_an_unknown_page_cancels_the_whole_removal(service):
    # 一部だけ消えて残りが黙って無視されるのが一番たちが悪い。何も消えないことまで確かめる
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(3)]
    pack = service.packs.create("Unknown page", event_ids)

    with pytest.raises(packs_module.PackError):
        service.packs.remove_pages(pack["id"], [0, 99])

    unchanged = service.packs.get(pack["id"])
    assert {item["event_id"] for item in unchanged["items"]} == set(event_ids)
    assert unchanged["page_count"] == pack["page_count"]


def test_removing_every_page_is_rejected(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(2)]
    pack = service.packs.create("All pages", event_ids)
    positions = [page["position"] for page in pack["pages"]]

    with pytest.raises(packs_module.PackError):
        service.packs.remove_pages(pack["id"], positions)
    assert len(service.packs.get(pack["id"])["items"]) == len(event_ids)


def test_revoked_document_cannot_be_edited(service):
    event_ids = [add_event(service, f"slide {index}", seconds=index) for index in range(2)]
    pack = service.packs.create("Revoked", event_ids)
    service.packs.revoke(pack["id"])

    with pytest.raises(packs_module.PackError):
        service.packs.remove_pages(pack["id"], [0])


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


def test_html_has_compact_event_pages_and_technical_page(service):
    event_id = add_event(service, "重要な見出し\n100万円の効果\n次のアクション")
    pack = service.packs.create("Compact PDF", [event_id])
    document = service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8")
    assert "01 / 01" in document
    assert "重要OCR" in document
    assert "DOCUMENT INTEGRITY" in document
    assert "Manifest SHA-256" in document
    assert 'class="technical"' in document


def test_appendix_is_left_out_unless_requested(service):
    """付録は既定では付けない。

    付録は画面ページと同じ情報をもう一度並べるため、枚数が増えると文書の半分以上を
    占める(実測で105画面のとき236ページ中129ページ)。必要なときだけ付ける。
    """
    event_id = add_event(service, "議事録の本文\n次回の宿題")
    pack = service.packs.create("No appendix", [event_id])
    document = service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8")
    assert "抽出テキスト付録" not in document
    assert 'class="appendix"' not in document


def test_appendix_is_included_when_requested(service):
    event_id = add_event(service, "議事録の本文\n次回の宿題")
    pack = service.packs.create("With appendix", [event_id], include_appendix=True)
    document = service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8")
    assert "抽出テキスト付録" in document
    assert "議事録の本文" in document


def test_edge_generates_real_pdf_document(service):
    if not service.packs._edge_path():
        pytest.skip("Microsoft Edge is not available")
    event_id = add_event(service, "PDFのOCRテキスト")
    pack = service.packs.create("Real PDF", [event_id])
    target = service.packs.export_document(pack["id"], "pdf")
    assert target.read_bytes().startswith(b"%PDF")
    assert target.stat().st_size > 1000


# --- PDFのテキスト層の健全性 ---
#
# Edgeの--print-to-pdfは、埋め込んだフォントから「グリフ→文字」の逆引き表(ToUnicode)を
# 作る。游ゴシック系・メイリオは「一」(U+4E00)と康熙部首の「⼀」(U+2F00)を同じグリフで
# 持つため、逆引きで康熙部首側が選ばれてPDFのテキスト層が別文字になる。見た目は正常なので
# 目視では気づけず、PDF内検索・コピー・MCP経由でAIが読むテキストだけが壊れる。
# 自分のコードではなく外部ツールの出力が壊れるため、通常の単体テストでは検出できない。

# ToUnicodeが壊れると現れる文字の範囲(CJK部首補助 + 康熙部首)
_RADICAL_BLOCK = range(0x2E80, 0x2FE0)

# 実際に検証して不合格だった書体。再指定を防ぐために名前で弾く
_BROKEN_PDF_FONTS = ("Yu Gothic", "YuGothic", "Meiryo", "メイリオ")

# 康熙部首としても存在する常用漢字を含めた検証用テキスト
# (一 日 車 出 力 入 用 月 人 口 はいずれも部首の重複を持つ)
_ROUND_TRIP_TEXT = "一次規定の適用範囲は日本国内の車両出力と回転数に依存する"


def test_document_font_stack_avoids_fonts_that_break_the_pdf_text_layer(service):
    """文書テンプレートが、テキスト層を壊す書体を指定していないことを確認する。"""
    event_id = add_event(service, _ROUND_TRIP_TEXT)
    pack = service.packs.create("Font stack", [event_id])
    service.packs.approve(pack["id"])
    html_text = service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8")
    # CSSコメントには注意書きとして書体名そのものが出てくるため、判定前に取り除く
    declarations = re.sub(r"/\*.*?\*/", "", html_text, flags=re.DOTALL)

    assert "BIZ UDPGothic" in declarations
    for font in _BROKEN_PDF_FONTS:
        assert font not in declarations, f"{font} はPDFのテキスト層を壊すため使用できない"


def test_pdf_text_layer_round_trips_the_ocr_text(monkeypatch, service):
    """OCRテキストが、PDFのテキスト層から同じ文字として取り出せることを確認する。

    生成したPDFを読み直す往復テスト。テンプレートのフォント指定を変えたときに、
    テキスト層が壊れていないかをここで検出する。
    """
    if not service.packs._edge_path():
        pytest.skip("Microsoft Edge is not available")
    pypdf = pytest.importorskip("pypdf", reason="pypdf is required to read back the PDF")

    from visual_memory import mcp_server

    lines = [_ROUND_TRIP_TEXT]
    lines.extend(f"ROW_{index:02d}: result[{index}] = value < 100 && ready;" for index in range(1, 13))
    lines.append("LONG_ROW: " + "value += source[index]; " * 12 + "長行の末尾確認_END")
    event_id = add_event(service, "\n".join(lines))
    service.db.execute(
        "UPDATE evidence SET metadata_json=? WHERE event_id=? AND kind='ocr'",
        (json.dumps({"lines": [{"text": line, "confidence": .99, "polygon": []}
                               for line in lines]}), event_id),
    )
    pack = service.packs.create("Round trip", [event_id], include_appendix=True)
    monkeypatch.setattr(mcp_server, "_service", lambda: service.packs)
    manifest_text = json.loads(mcp_server.get_context_pack(pack["id"]))["items"][0]["ocr_text"]
    document = html.unescape(service.packs.export_document(pack["id"], "html").read_text(encoding="utf-8"))
    for line in lines:
        assert line in manifest_text
        assert line in document
    pdf_path = service.packs.export_document(pack["id"], "pdf")

    reader = pypdf.PdfReader(str(pdf_path))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
    # PDFは幅に応じて行を折り返すため、空白を除いた1本の文字列として突き合わせる
    flattened = "".join(extracted.split())

    radicals = sorted({character for character in extracted if ord(character) in _RADICAL_BLOCK})
    assert not radicals, f"テキスト層に部首ブロックの文字が混入している: {radicals}"
    for line in lines:
        assert "".join(line.split()) in flattened


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
