from __future__ import annotations

import base64
import ctypes
import html
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .db import Database
from .imaging import Redaction, apply_redactions, sha256_bytes, sha256_file
from .stitching import SourceFrame, render_document_pages

LOGGER = logging.getLogger(__name__)


def _set_windows_dll_directory(path: str | None) -> None:
    if not ctypes.windll.kernel32.SetDllDirectoryW(path):
        raise OSError(ctypes.get_last_error(), "SetDllDirectoryW failed")


@contextmanager
def _external_program_environment():
    """Give system-installed programs the normal Windows DLL search path.

    PyInstaller changes the process-wide DLL directory to the bundle. External
    programs inherit that directory and can load incompatible bundled DLLs.
    Restore the PyInstaller directory immediately after process creation.
    """
    environment = dict(os.environ)
    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        yield environment
        return

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    _set_windows_dll_directory(None)
    try:
        yield environment
    finally:
        _set_windows_dll_directory(str(bundle_root))


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _display_time(value: str | None) -> str:
    if not value:
        return "-"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        local = parsed.astimezone()
        timezone_name = "JST" if local.utcoffset() == timedelta(hours=9) else local.tzname()
        return f"{local:%Y/%m/%d %H:%M:%S} {timezone_name or ''}".rstrip()
    except ValueError:
        return value


_NOISE_LABELS = {
    "検索", "q 検索", "ファイル", "ホーム", "挿入", "描画", "デザイン",
    "画面切り替え", "アニメーション", "スライドショー", "record", "校閲",
    "表示", "ヘルプ", "acrobat", "共有", "日本語", "ノート", "表示設定",
    "コメント", "アクセシビリティ：検討が必要です",
    "アクセシビリティ: 検討が必要です",
}
_FILE_TITLE = re.compile(
    r"(?i)(.+?\.(?:pptx?|docx?|xlsx?|pdf|py|ipynb|js|ts|tsx|jsx|java|cs|cpp|c|h))"
)


def _normalize_ocr_line(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _looks_like_chrome_noise(value: str) -> bool:
    normalized = _normalize_ocr_line(value)
    lowered = normalized.casefold()
    if not normalized or lowered in _NOISE_LABELS:
        return True
    if len(normalized) <= 1 and not normalized.isalpha():
        return True
    if re.fullmatch(r"[+×□○●△▽◇◆★☆※*%]+", normalized):
        return True
    ui_terms = (
        "ファイル", "ホーム", "挿入", "描画", "デザイン", "画面切り替え",
        "アニメーション", "スライドショー", "record", "校閲", "表示", "ヘルプ",
        "acrobat", "共有", "ノート", "表示設定",
    )
    if sum(term in lowered for term in ui_terms) >= 2:
        return True
    if _FILE_TITLE.search(normalized) and any(
        app in lowered for app in ("powerpoint", "word", "excel", "chrome", "edge")
    ):
        return True
    if re.fullmatch(r"\d+[*/]\d+", normalized):
        return True
    if re.fullmatch(r"スライド\s*\d+/\d+", normalized):
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", normalized):
        return True
    if re.fullmatch(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", normalized):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?%", normalized):
        return True
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?\s*℃", normalized))


def _automatic_title(rows: list[Any]) -> str:
    candidates: list[str] = []
    file_titles: list[str] = []
    for row in rows:
        for raw in str(row["ocr_text"] or "").splitlines():
            line = _normalize_ocr_line(raw)
            match = _FILE_TITLE.search(line)
            if match:
                title = re.sub(
                    r"(?i)\s*[-–—]\s*(PowerPoint|Word|Excel|Adobe Acrobat|Google Chrome|Microsoft Edge).*$",
                    "",
                    match.group(1),
                ).strip()
                title = re.sub(r"(?<=[ァ-ヶ])-+(?=[ァ-ヶ])", "ー", title)
                title = re.sub(
                    r"(?i)\.(?:pptx?|docx?|xlsx?|pdf|py|ipynb|js|ts|tsx|jsx|java|cs|cpp|c|h)$",
                    "",
                    title,
                )
                if title:
                    file_titles.append(title)
            if 5 <= len(line) <= 100 and not _looks_like_chrome_noise(line):
                candidates.append(line)
    if file_titles:
        counts = Counter(file_titles)
        simplified_penalty = set("值视图组")
        return max(
            file_titles,
            key=lambda value: (
                -sum(character in simplified_penalty for character in value),
                counts[value],
                len(value),
            ),
        )
    if candidates:
        counts = Counter(candidates)
        return max(candidates, key=lambda value: (counts[value], len(value)))
    first_time = rows[0]["started_at"] if rows else None
    return f"画面コンテキスト {_display_time(first_time)}"


def _safe_ocr_lines(
    metadata_json: str | None,
    original_text: str,
    redactions: list[Redaction],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    try:
        metadata_lines = json.loads(metadata_json or "{}").get("lines", [])
    except (json.JSONDecodeError, AttributeError):
        metadata_lines = []
    if not metadata_lines:
        if redactions:
            return []
        return [
            {"text": line, "polygon": []}
            for line in original_text.splitlines()
            if _normalize_ocr_line(line)
        ]

    safe_lines: list[dict[str, Any]] = []
    for line in metadata_lines:
        text = _normalize_ocr_line(str(line.get("text", "")))
        polygon = line.get("polygon") or []
        if not text:
            continue
        xs = [float(point[0]) for point in polygon if len(point) >= 2]
        ys = [float(point[1]) for point in polygon if len(point) >= 2]
        if redactions and (not xs or not ys):
            continue
        touched = False
        if xs and ys:
            line_left, line_right = min(xs), max(xs)
            line_top, line_bottom = min(ys), max(ys)
            for raw in redactions:
                item = raw.normalized()
                if (
                    line_left < (item.x + item.width) * width
                    and line_right > item.x * width
                    and line_top < (item.y + item.height) * height
                    and line_bottom > item.y * height
                ):
                    touched = True
                    break
        if not touched:
            safe_lines.append({"text": text, "polygon": polygon})
    return safe_lines


def _clean_ocr_items(
    line_groups: list[list[dict[str, Any]]], dimensions: list[tuple[int, int]]
) -> list[str]:
    document_frequency: Counter[str] = Counter()
    for lines in line_groups:
        document_frequency.update({_normalize_ocr_line(line["text"]).casefold() for line in lines})
    repeat_threshold = max(2, math.ceil(len(line_groups) * 0.6))
    results: list[str] = []
    for lines, (_width, height) in zip(line_groups, dimensions, strict=True):
        cleaned: list[str] = []
        seen: set[str] = set()
        for line in lines:
            text = _normalize_ocr_line(line["text"])
            key = text.casefold()
            if key in seen or _looks_like_chrome_noise(text):
                continue
            polygon = line.get("polygon") or []
            ys = [float(point[1]) for point in polygon if len(point) >= 2]
            repeated_outer_chrome = False
            if ys and height > 0 and document_frequency[key] >= repeat_threshold:
                center_y = (min(ys) + max(ys)) / 2 / height
                repeated_outer_chrome = center_y <= 0.14 or center_y >= 0.82
            if repeated_outer_chrome:
                continue
            seen.add(key)
            cleaned.append(text)
        results.append("\n".join(cleaned))
    return results


def _important_ocr(value: str, limit: int = 8) -> list[str]:
    important: list[str] = []
    for raw in value.splitlines():
        line = _normalize_ocr_line(raw)
        if not line or _looks_like_chrome_noise(line):
            continue
        if len(line) > 140:
            line = f"{line[:137]}..."
        important.append(line)
        if len(important) >= limit:
            break
    return important


def _ocr_after_redactions(
    metadata_json: str | None,
    original_text: str,
    redactions: list[Redaction],
    width: int,
    height: int,
) -> str:
    """Remove complete OCR lines touched by a destructive image redaction.

    If coordinate evidence is missing, fail closed and omit OCR text instead of
    risking that text hidden in the image remains available through MCP.
    """
    return "\n".join(
        line["text"]
        for line in _safe_ocr_lines(
            metadata_json, original_text, redactions, width, height
        )
    )


class PackError(ValueError):
    pass


class ContextPackService:
    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings

    def artifact_dir(self, pack: dict[str, Any] | str) -> Path:
        value = self.get(pack, include_items=False) if isinstance(pack, str) else pack
        artifact_path = value.get("artifact_path")
        return Path(artifact_path) if artifact_path else self.settings.packs_dir / value["id"]

    def _remove_artifact_dir(self, path: Path, pack_id: str) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(self.settings.packs_dir.resolve())
            if not resolved.name.startswith(pack_id):
                return
            shutil.rmtree(resolved)
        except (OSError, ValueError):
            LOGGER.warning("Unable to remove superseded context document at %s", path)

    def create(
        self,
        title: str,
        event_ids: Iterable[int],
        query: str = "",
        note: str = "",
        deduplicate_overlaps: bool = True,
    ) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(int(value) for value in event_ids))
        if not unique_ids:
            raise PackError("Select at least one event")
        if len(unique_ids) > self.settings.max_pack_images:
            raise PackError(
                f"A context document can contain at most {self.settings.max_pack_images} images"
            )
        placeholders = ",".join("?" for _ in unique_ids)
        fetched = self.db.fetchall(
            f"SELECT * FROM screen_event WHERE id IN ({placeholders})", unique_ids
        )
        rows_by_id = {int(row["id"]): row for row in fetched}
        rows = [rows_by_id[event_id] for event_id in unique_ids if event_id in rows_by_id]
        if len(rows) != len(unique_ids):
            raise PackError("One or more selected events no longer exist")
        pack_id = str(uuid.uuid4())
        created = iso(utcnow())
        document_title = title.strip() or _automatic_title(rows)
        with self.db.transaction() as connection:
            connection.execute(
                """INSERT INTO context_pack(
                       id,title,query,note,deduplicate_overlaps,status,created_at
                   ) VALUES(?,?,?,?,?, 'draft', ?)""",
                (
                    pack_id,
                    document_title,
                    query.strip(),
                    note.strip(),
                    int(deduplicate_overlaps),
                    created,
                ),
            )
            for position, row in enumerate(rows):
                connection.execute(
                    """INSERT INTO context_pack_item(pack_id,event_id,position,ocr_text)
                       VALUES(?,?,?,?)""",
                    (pack_id, row["id"], position, row["ocr_text"]),
                )
        return self.approve(pack_id)

    def list(self, include_drafts: bool = True, search: str = "") -> list[dict[str, Any]]:
        self.expire_due()
        conditions: list[str] = []
        params: list[Any] = []
        if not include_drafts:
            conditions.append("status='approved' AND expires_at > ?")
            params.append(iso(utcnow()))
        if search:
            conditions.append("(title LIKE ? OR note LIKE ? OR query LIKE ?)")
            value = f"%{search}%"
            params.extend((value, value, value))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = self.db.fetchall(
            f"""SELECT p.*, COUNT(i.event_id) AS item_count
                FROM context_pack p LEFT JOIN context_pack_item i ON i.pack_id=p.id
                {where} GROUP BY p.id ORDER BY p.created_at DESC""",
            params,
        )
        return [dict(row) for row in rows]

    def get(self, pack_id: str, include_items: bool = True) -> dict[str, Any]:
        self.expire_due()
        pack = self.db.fetchone("SELECT * FROM context_pack WHERE id=?", (pack_id,))
        if not pack:
            raise PackError("Context pack not found")
        result = dict(pack)
        if include_items:
            items = self.db.fetchall(
                """
                SELECT i.*, e.started_at, e.ended_at, e.frame_path, e.thumbnail_path,
                       e.session_id, e.width, e.height, e.event_kind,
                       e.ocr_text AS source_ocr_text
                FROM context_pack_item i JOIN screen_event e ON e.id=i.event_id
                WHERE i.pack_id=? ORDER BY i.position
                """,
                (pack_id,),
            )
            result["items"] = [dict(item) for item in items]
        return result

    def get_approved(self, pack_id: str) -> dict[str, Any]:
        pack = self.get(pack_id, include_items=True)
        if pack["status"] != "approved" or not pack["expires_at"] or pack["expires_at"] <= iso(utcnow()):
            raise PermissionError("Context pack is not approved or has expired")
        return pack

    def approve(self, pack_id: str, expiry_hours: int | None = None) -> dict[str, Any]:
        pack = self.get(pack_id, include_items=True)
        if pack["status"] not in {"draft", "approved"}:
            raise PackError("Only active context documents can be rebuilt")
        hours = expiry_hours or self.settings.default_approval_hours
        if hours < 1 or hours > 24 * 365:
            raise PackError("Approval duration must be between 1 hour and 365 days")
        previous_dir = self.artifact_dir(pack)
        build_id = uuid.uuid4().hex[:12]
        pack_dir = self.settings.packs_dir / f".{pack_id}-{build_id}.tmp"
        final_dir = self.settings.packs_dir / f"{pack_id}-{build_id}"
        images_dir = pack_dir / "images"
        images_dir.mkdir(parents=True)
        self.db.execute(
            "UPDATE context_pack SET status='draft', build_error=NULL WHERE id=?",
            (pack_id,),
        )
        try:
            sources: list[SourceFrame] = []
            source_rows: dict[int, dict[str, Any]] = {}
            for item in pack["items"]:
                redactions = [Redaction(**value) for value in json.loads(item["redactions_json"])]
                ocr_evidence = self.db.fetchone(
                    """SELECT metadata_json FROM evidence
                       WHERE event_id=? AND kind='ocr' ORDER BY id DESC LIMIT 1""",
                    (item["event_id"],),
                )
                safe_lines = _safe_ocr_lines(
                    ocr_evidence["metadata_json"] if ocr_evidence else None,
                    item["source_ocr_text"],
                    redactions,
                    int(item["width"]),
                    int(item["height"]),
                )
                with Image.open(item["frame_path"]) as source:
                    image = apply_redactions(source, redactions).convert("RGB")
                event_id = int(item["event_id"])
                redaction_values = json.loads(item["redactions_json"])
                sources.append(
                    SourceFrame(
                        event_id=event_id,
                        session_id=str(item["session_id"]),
                        started_at=str(item["started_at"]),
                        ended_at=str(item["ended_at"]),
                        image=image,
                        lines=safe_lines,
                        redactions=redaction_values,
                    )
                )
                source_rows[event_id] = {
                    "event_id": event_id,
                    "started_at": item["started_at"],
                    "ended_at": item["ended_at"],
                    "redactions": redaction_values,
                }

            rendered_pages = render_document_pages(
                sources, bool(pack.get("deduplicate_overlaps", 0))
            )
            cleaned_ocr = _clean_ocr_items(
                [page.lines for page in rendered_pages],
                [(page.image.width, page.image.height) for page in rendered_pages],
            )
            manifest_items: list[dict[str, Any]] = []
            for position, (page, ocr_text) in enumerate(
                zip(rendered_pages, cleaned_ocr, strict=True)
            ):
                filename = f"{position + 1:02d}-page-event-{page.event_id}.webp"
                target = images_dir / filename
                page.image.save(target, "WEBP", quality=self.settings.webp_quality, method=4)
                digest = sha256_file(target)
                anchor = source_rows[page.event_id]
                manifest_items.append(
                    {
                        "event_id": page.event_id,
                        "position": position,
                        "started_at": page.started_at,
                        "ended_at": page.ended_at,
                        "image": f"images/{filename}",
                        "sha256": digest,
                        "redactions": anchor["redactions"],
                        "source_event_ids": page.source_event_ids,
                        "source_events": [
                            source_rows[event_id] for event_id in page.source_event_ids
                        ],
                        "stitch": page.stitch,
                        "ocr_text": ocr_text,
                        "important_ocr": _important_ocr(ocr_text),
                    }
                )

            if (
                pack["status"] == "approved"
                and pack.get("approved_at")
                and pack.get("expires_at")
                and expiry_hours is None
            ):
                # expiry_hoursが明示されていない再ビルドは、既存の承認日時・期限を維持する
                approved_at = datetime.fromisoformat(pack["approved_at"])
                expires_at = datetime.fromisoformat(pack["expires_at"])
            elif pack["status"] == "approved" and expiry_hours is not None:
                # 明示的な再承認は現在時刻から新しい承認期間を開始する
                approved_at = utcnow()
                expires_at = approved_at + timedelta(hours=hours)
            else:
                approved_at = utcnow()
                expires_at = approved_at + timedelta(hours=hours)
            manifest = {
                "schema_version": 3,
                "id": pack_id,
                "title": pack["title"],
                "query": pack["query"],
                "note": pack["note"],
                "deduplicate_overlaps": bool(pack.get("deduplicate_overlaps", 0)),
                "approved_at": iso(approved_at),
                "expires_at": iso(expires_at),
                "items": manifest_items,
            }
            manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            manifest_sha = sha256_bytes(manifest_bytes)
            (pack_dir / "manifest.json").write_bytes(manifest_bytes)
            (pack_dir / "context.md").write_text(self._markdown(manifest), encoding="utf-8")
            (pack_dir / "context.html").write_text(
                self._single_file_html(manifest, pack_dir, manifest_sha), encoding="utf-8"
            )

            event_ocr: dict[int, str] = {}
            for event_id in source_rows:
                groups: list[list[dict[str, Any]]] = []
                dimensions: list[tuple[int, int]] = []
                for page in rendered_pages:
                    owned_lines = [line for line in page.lines if line.get("event_id") == event_id]
                    if owned_lines:
                        groups.append(owned_lines)
                        dimensions.append((page.image.width, page.image.height))
                event_ocr[event_id] = (
                    "\n".join(value for value in _clean_ocr_items(groups, dimensions) if value)
                    if groups
                    else ""
                )

            pack_dir.replace(final_dir)
            try:
                with self.db.transaction() as connection:
                    for event_id in source_rows:
                        output = next(
                            (
                                item
                                for item in manifest_items
                                if event_id in item["source_event_ids"]
                            ),
                            None,
                        )
                        connection.execute(
                            """UPDATE context_pack_item SET image_path=?, sha256=?, ocr_text=?
                               WHERE pack_id=? AND event_id=?""",
                            (
                                str(final_dir / output["image"]) if output else None,
                                output["sha256"] if output else None,
                                event_ocr[event_id],
                                pack_id,
                                event_id,
                            ),
                        )
                    connection.execute(
                        """UPDATE context_pack SET status='approved', approved_at=?, expires_at=?,
                                  manifest_sha256=?, artifact_path=?, build_error=NULL WHERE id=?""",
                        (
                            iso(approved_at),
                            iso(expires_at),
                            manifest_sha,
                            str(final_dir),
                            pack_id,
                        ),
                    )
            except Exception:
                self._remove_artifact_dir(final_dir, pack_id)
                raise
            if previous_dir != final_dir and previous_dir.exists():
                self._remove_artifact_dir(previous_dir, pack_id)
            return self.get(pack_id, include_items=True)
        except Exception as exc:
            if pack_dir.exists():
                shutil.rmtree(pack_dir, ignore_errors=True)
            self.db.execute(
                "UPDATE context_pack SET status='draft', build_error=? WHERE id=?",
                (str(exc)[:2000], pack_id),
            )
            raise

    def revoke(self, pack_id: str) -> dict[str, Any]:
        pack = self.get(pack_id, include_items=False)
        if pack["status"] not in ("approved", "expired"):
            raise PackError("Only approved or expired packs can be revoked")
        self.db.execute(
            "UPDATE context_pack SET status='revoked', revoked_at=? WHERE id=?",
            (iso(utcnow()), pack_id),
        )
        return self.get(pack_id, include_items=True)

    def expire_due(self) -> int:
        now = iso(utcnow())
        with self.db.connect() as connection:
            cursor = connection.execute(
                "UPDATE context_pack SET status='expired' WHERE status='approved' AND expires_at <= ?", (now,)
            )
            connection.commit()
            return cursor.rowcount

    def export_document(self, pack_id: str, document_format: str = "pdf") -> Path:
        pack = self.get(pack_id, include_items=True)
        if pack["status"] not in ("approved", "expired", "revoked"):
            raise PackError("Approve the pack before exporting it")
        document_format = document_format.lower().strip()
        if document_format not in {"html", "pdf"}:
            raise PackError("Document format must be html or pdf")
        pack_dir = self.artifact_dir(pack)
        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            raise PackError("Pack files are missing")
        html_path = pack_dir / "context.html"
        if not html_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            html_path.write_text(
                self._single_file_html(manifest, pack_dir, pack["manifest_sha256"] or ""),
                encoding="utf-8",
            )
        if document_format == "html":
            return html_path
        target = pack_dir / "context.pdf"
        if not target.exists() or target.stat().st_mtime < html_path.stat().st_mtime:
            self._render_pdf(html_path, target)
        return target

    def get_approved_document(self, pack_id: str, document_format: str = "pdf") -> Path:
        self.get_approved(pack_id)
        return self.export_document(pack_id, document_format)

    @staticmethod
    def _single_file_html(manifest: dict[str, Any], pack_dir: Path, manifest_sha: str) -> str:
        sections: list[str] = []
        appendix_sections: list[str] = []
        total = len(manifest["items"])
        for index, item in enumerate(manifest["items"], start=1):
            image_path = pack_dir / item["image"]
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            raw_ocr = item.get("ocr_text", "")
            ocr_text = html.escape(raw_ocr)
            important = item.get("important_ocr") or _important_ocr(raw_ocr)
            important_html = "".join(f"<li>{html.escape(line)}</li>" for line in important)
            if not important_html:
                important_html = "<li class=\"muted-line\">文字情報はありません</li>"
            display_time = html.escape(_display_time(item.get("started_at")))
            sections.append(
                f"""
                <section class="event">
                  <header><strong>{index:02d} / {total:02d}</strong><span>{display_time}</span></header>
                  <img src="data:image/webp;base64,{image_data}" alt="Captured screen {index}">
                  <div class="important-ocr"><h2>重要OCR</h2><ul>{important_html}</ul></div>
                  <details><summary>抽出テキスト全文</summary><pre>{ocr_text}</pre></details>
                </section>
                """
            )
            appendix_sections.append(
                f"""
                <article class="appendix-item">
                  <h3>{index:02d} / {total:02d} <span>{display_time}</span></h3>
                  <pre>{ocr_text or '文字情報はありません'}</pre>
                </article>
                """
            )
        title = html.escape(manifest["title"])
        note = html.escape(manifest.get("note", ""))
        query = html.escape(manifest.get("query", ""))
        approved_display = html.escape(_display_time(manifest.get("approved_at")))
        expires_display = html.escape(_display_time(manifest.get("expires_at")))
        return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta http-equiv="Content-Security-Policy"
        content="default-src 'none'; img-src data:; style-src 'unsafe-inline'">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: "Yu Gothic UI", "Meiryo", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #080b0d; color: #edf2f3; }}
    .cover {{
      min-height: 100vh; padding: 8vh 7vw; display: grid;
      align-content: center; background: #0d1215;
    }}
    .eyebrow {{ color: #9cff38; font: 700 12px ui-monospace, monospace; letter-spacing: .12em; }}
    h1 {{ margin: 12px 0 24px; font-size: clamp(32px, 6vw, 76px); line-height: 1.05; }}
    .meta {{ color: #aeb9bc; line-height: 1.7; }}
    .note {{ max-width: 70ch; white-space: pre-wrap; }}
    .event {{
      padding: 28px; min-height: 100vh; border-top: 1px solid #273136;
      break-after: page; page-break-after: always;
    }}
    .event header {{
      display: flex; justify-content: space-between; gap: 16px;
      margin-bottom: 12px; color: #9cff38;
    }}
    .event img {{ display: block; width: 100%; max-height: 68vh; object-fit: contain; background: #000; }}
    .important-ocr {{ margin-top: 12px; }}
    .important-ocr h2 {{ margin: 0 0 6px; color: #aeb9bc; font-size: 12px; letter-spacing: .06em; }}
    .important-ocr ul {{
      margin: 0; padding-left: 20px; columns: 2; column-gap: 30px;
      font-size: 12px; line-height: 1.35;
    }}
    .important-ocr li {{ break-inside: avoid; margin-bottom: 2px; }}
    .muted-line {{ color: #7c878c; }}
    details {{ margin-top: 14px; }}
    summary {{ color: #aeb9bc; cursor: pointer; }}
    pre {{
      white-space: pre-wrap; overflow-wrap: anywhere;
      font: 12px/1.45 "Yu Gothic UI", "Meiryo", sans-serif; color: #d8e0e2;
    }}
    .appendix, .technical {{ padding: 36px; }}
    .appendix > h2, .technical > h2 {{ margin: 0 0 20px; font-size: 28px; }}
    .appendix-item {{ padding: 14px 0; border-top: 1px solid #273136; }}
    .appendix-item h3 {{ margin: 0 0 8px; color: #9cff38; font-size: 13px; }}
    .appendix-item h3 span {{ float: right; font-weight: 400; }}
    .appendix-item pre {{ margin: 0; }}
    .technical {{ min-height: 100vh; display: grid; align-content: center; }}
    .technical code {{ overflow-wrap: anywhere; }}
    @page {{ size: A4 landscape; margin: 9mm; }}
    @media print {{
      :root {{ color-scheme: light; }} body {{ background: white; color: #111; }}
      .cover {{ min-height: 180mm; background: white; break-after: page; page-break-after: always; }}
      .event {{ padding: 0; height: 180mm; min-height: 180mm; border: 0; overflow: hidden; }}
      .event img {{ height: 132mm; max-height: 132mm; }}
      .event .important-ocr {{ height: 30mm; overflow: hidden; }}
      .event details {{ display: none; }}
      .appendix {{ break-before: page; page-break-before: always; padding: 0; }}
      .appendix-item {{ break-inside: auto; page-break-inside: auto; }}
      .appendix-item pre {{ font-size: 9px; line-height: 1.25; }}
      .technical {{ min-height: 180mm; padding: 0; break-before: page; page-break-before: always; }}
      .meta, summary, pre {{ color: #333; }}
    }}
  </style>
</head>
<body>
  <section class="cover">
    <p class="eyebrow">EXTERNAL PC VISUAL MEMORY</p>
    <h1>{title}</h1>
    <div class="meta">
      <p>
        作成: {approved_display}<br>
        MCP利用期限: {expires_display}<br>
        画面数: {len(manifest['items'])}
      </p>
      {f'<p>Search: {query}</p>' if query else ''}
      {f'<p class="note">{note}</p>' if note else ''}
    </div>
  </section>
  {''.join(sections)}
  <section class="appendix">
    <h2>抽出テキスト付録</h2>
    <p class="meta">画面上で反復するメニューや時刻などを除外したOCR全文です。</p>
    {''.join(appendix_sections)}
  </section>
  <section class="technical">
    <div>
      <p class="eyebrow">DOCUMENT INTEGRITY</p>
      <h2>技術情報</h2>
      <p class="meta">
        Document ID: {html.escape(manifest['id'])}<br>
        Manifest SHA-256: <code>{html.escape(manifest_sha)}</code><br>
        Created (UTC): {html.escape(manifest['approved_at'])}<br>
        Expires (UTC): {html.escape(manifest['expires_at'])}
      </p>
    </div>
  </section>
</body>
</html>
"""

    @staticmethod
    def _edge_path() -> Path | None:
        candidates = [
            shutil.which("msedge"),
            os.path.join(
                os.environ.get("PROGRAMFILES(X86)", ""),
                "Microsoft", "Edge", "Application", "msedge.exe",
            ),
            os.path.join(
                os.environ.get("PROGRAMFILES", ""),
                "Microsoft", "Edge", "Application", "msedge.exe",
            ),
        ]
        return next((Path(value) for value in candidates if value and Path(value).exists()), None)

    @classmethod
    def _render_pdf(cls, html_path: Path, target: Path) -> None:
        edge = cls._edge_path()
        if not edge:
            raise PackError("Microsoft Edge is required to create a PDF document")
        profile = html_path.parent / f".edge-pdf-profile-{uuid.uuid4().hex}"
        command = [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={target}",
            html_path.resolve().as_uri(),
        ]
        try:
            with _external_program_environment() as environment:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                )
            try:
                _stdout, stderr = process.communicate(timeout=120)
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.communicate()
                raise PackError("Microsoft Edge timed out while creating the PDF") from error
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not target.exists():
                time.sleep(0.2)
            if process.returncode or not target.exists() or target.stat().st_size == 0:
                detail = stderr.decode(errors="replace")[-1000:]
                raise PackError(f"Unable to create PDF document: {detail or process.returncode}")
        finally:
            shutil.rmtree(profile, ignore_errors=True)

    @staticmethod
    def _markdown(manifest: dict[str, Any]) -> str:
        lines = [
            f"# {manifest['title']}",
            "",
            f"Approved: {manifest['approved_at']}",
            f"Expires: {manifest['expires_at']}",
            "",
        ]
        if manifest.get("query"):
            lines += [f"Search: {manifest['query']}", ""]
        if manifest.get("note"):
            lines += [manifest["note"], ""]
        for item in manifest["items"]:
            lines += [
                f"## Event {item['event_id']} · {item['started_at']}",
                "",
                f"![event {item['event_id']}]({item['image']})",
                "",
                "```text",
                item["ocr_text"],
                "```",
                "",
            ]
        return "\n".join(lines)
