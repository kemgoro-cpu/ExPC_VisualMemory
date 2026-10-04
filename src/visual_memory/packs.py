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
import tempfile
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from PIL import Image

from .config import Settings
from .db import Database
from .imaging import Redaction, apply_redactions, sha256_bytes, sha256_file
from .stitching import SourceFrame, render_document_pages
from .textnorm import (
    FILE_TITLE_PATTERN,
    TITLE_EXTENSIONS,
    filter_ocr_lines,
    line_key,
    looks_like_chrome_noise,
    normalize_ocr_line,
)

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


# ファイル名らしき部分の直後が日本語の続き(ひらがな・カタカナ・漢字)であれば、
# 「rendererではNode.jsを無効にし」のような文中の言及であってファイル名の提示では
# ないと判断する
_MID_SENTENCE_CONTINUATION = re.compile(r"[ぁ-んァ-ヶー一-龠]")


def _automatic_title(rows: list[Any]) -> str:
    candidates: list[str] = []
    file_titles: list[str] = []
    for row in rows:
        for raw in str(row["ocr_text"] or "").splitlines():
            line = normalize_ocr_line(raw)
            match = FILE_TITLE_PATTERN.search(line)
            trailing = line[match.end() :] if match else ""
            if match and not _MID_SENTENCE_CONTINUATION.match(trailing):
                title = re.sub(
                    r"(?i)\s*[-–—]\s*(PowerPoint|Word|Excel|Adobe Acrobat|Google Chrome|Microsoft Edge).*$",
                    "",
                    match.group(1),
                ).strip()
                title = re.sub(r"(?<=[ァ-ヶ])-+(?=[ァ-ヶ])", "ー", title)
                title = re.sub(rf"(?i)\.(?:{TITLE_EXTENSIONS})$", "", title)
                # パンくずリストやタブのパス表示(「C: > Users > … > PLAN.md」)から
                # ファイル名部分だけを取り出す
                title = re.split(r"\s*(?:>|\\|/)\s*", title)[-1].strip()
                if title:
                    file_titles.append(title)
            if 5 <= len(line) <= 100 and not looks_like_chrome_noise(line):
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
            if normalize_ocr_line(line)
        ]

    safe_lines: list[dict[str, Any]] = []
    for line in metadata_lines:
        text = normalize_ocr_line(str(line.get("text", "")))
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
            # confidence は重要行の選定で「誤認識らしさ」の判定に使うため引き継ぐ
            safe_lines.append(
                {"text": text, "polygon": polygon, "confidence": line.get("confidence")}
            )
    return safe_lines


def _clean_ocr_line_groups(
    line_groups: list[list[dict[str, Any]]], dimensions: list[tuple[int, int]]
) -> list[list[dict[str, Any]]]:
    """ページごとの行データからノイズを除き、行データのまま返す。"""
    # ページ群全体で何回出てきた行かを数え、6割以上のページに出る行を
    # 「常時表示のUI」の候補とする。実際に落とすかは位置も見て決める(filter_ocr_lines)
    document_frequency: Counter[str] = Counter()
    for lines in line_groups:
        document_frequency.update({line_key(line["text"]) for line in lines})
    repeat_threshold = max(2, math.ceil(len(line_groups) * 0.6))
    repeated_keys = {key for key, count in document_frequency.items() if count >= repeat_threshold}
    return [
        filter_ocr_lines(lines, height, repeated_keys)
        for lines, (_width, height) in zip(line_groups, dimensions, strict=True)
    ]


def _clean_ocr_items(
    line_groups: list[list[dict[str, Any]]], dimensions: list[tuple[int, int]]
) -> list[str]:
    """ページごとのノイズ除去済みテキストを返す(:func:`_clean_ocr_line_groups` のテキスト版)。"""
    return [
        "\n".join(line["text"] for line in lines)
        for lines in _clean_ocr_line_groups(line_groups, dimensions)
    ]


# --- 重要OCRの選定 ---
#
# 1画面から抜き出した行のうち、その画面を代表する数行を選んで文書に載せる。
# 以前は先頭から順に必要数を取るだけだったため、画面の上のほうにあるアイコンや
# 飾りの誤認識(「乙〇0」など)がそのまま先頭に並んでいた。
#
# 選び方は「画面の上から順に読む」を基本にする。画面の上部にはたいていタイトルや
# 節見出しが来るうえ、OCRの1行は文の断片なので、離れた行を拾い集めると文脈が壊れるため。
# そのうえで次の2点を加味する:
#   1. 文字の大きさ - 本文より明らかに大きい行は見出しとみなし、下のほうにあっても拾う
#   2. OCRの確信度  - アイコンや飾りの誤認識は確信度が低く出やすいので下げる
#
# 選んだあとは点数順ではなく元の並び順に戻す。点数順のままだと見出しと本文が
# 入れ替わって文脈が壊れるため。
#
# 補足: 「長い行ほど重要」という重み付けも試したが、2段組みの本文断片が見出しを
# 押しのけてしまい、かえって要約として読みにくくなったため採用していない。

# 意味を持つ文字(かな・漢字・英字)。記号・数字・幾何学模様はここに含めない
_MEANINGFUL_CHARACTER = re.compile(r"[ぁ-んァ-ヶ一-龠a-zA-Z]")

# これを下回る確信度の行は誤認識とみなして捨てる
_MIN_LINE_CONFIDENCE = 0.55

# 飾りとみなす長さの上限。これ以下の短い行だけを飾り判定の対象にする
_DECORATION_MAX_LENGTH = 4

# 重要OCRに載せる1行の最大文字数
_IMPORTANT_LINE_MAX_LENGTH = 140

# 本文(ページ内の中央値)の何倍の高さなら見出しとみなすか
_HEADING_HEIGHT_RATIO = 1.3

# 見出しに与える加点。読み順の重み(_ORDER_WEIGHT)と同程度にしてあるので、
# 画面のいちばん下にある見出しが、いちばん上の本文とほぼ同じ強さで拾われる
_HEADING_BONUS = 2.0

# 読み順の重み。画面の先頭を+_ORDER_WEIGHT、末尾を0として線形に配分する
_ORDER_WEIGHT = 2.0


def _looks_like_decoration(text: str) -> bool:
    """アイコンや飾り罫の誤認識に見える短い行かを判定する。

    「乙〇0」「@]」「100」のように、短いうえに意味のある文字がほとんど無い行が対象。
    「結論」「要旨」「第1章」のような短い見出しは残す必要があるため、
    意味のある文字が2つ以上あれば飾りとはみなさない。
    """
    if len(text) > _DECORATION_MAX_LENGTH:
        return False
    return len(_MEANINGFUL_CHARACTER.findall(text)) < 2


def _line_text_height(polygon: Any) -> float:
    """行を囲む多角形の高さ(px)を返す。座標が無ければ0。"""
    ys = [float(point[1]) for point in polygon or [] if len(point) >= 2]
    return max(ys) - min(ys) if len(ys) >= 2 else 0.0


def _important_ocr(
    lines: Iterable[Mapping[str, Any]] | str, limit: int = 8
) -> list[str]:
    """1画面分の行データから、代表となる行を最大 ``limit`` 件選ぶ。

    Args:
        lines: ``{"text", "polygon", "confidence"}`` 形式の行データ。
            文字列を渡すと1行1要素として扱う(``polygon``/``confidence`` を持たない
            古いmanifestからの再生成用。この場合は文字数と文字種だけで判定する)。
        limit: 選ぶ行数の上限。

    Returns:
        元の並び順に戻した行テキストの一覧。
    """
    if isinstance(lines, str):
        items: list[Mapping[str, Any]] = [{"text": raw} for raw in lines.splitlines()]
    else:
        items = list(lines)

    # 1. 明らかに本文でない行を落とす
    #    残った行には0から順に通し番号を振る。これが「画面の上からの読み順」になり、
    #    点数付けと、最後に元の並びへ戻すときの両方で使う
    candidates: list[tuple[int, str, float, float | None]] = []
    for line in items:
        text = normalize_ocr_line(str(line.get("text", "")))
        if not text or looks_like_chrome_noise(text) or _looks_like_decoration(text):
            continue
        confidence = line.get("confidence")
        confidence = float(confidence) if isinstance(confidence, int | float) else None
        if confidence is not None and confidence < _MIN_LINE_CONFIDENCE:
            continue
        candidates.append(
            (len(candidates), text, _line_text_height(line.get("polygon")), confidence)
        )
    if not candidates:
        return []

    # 2. 点数を付ける。文字の大きさはページ内の中央値を基準にした比で見る
    #    (解像度や画面ごとの文字サイズの違いに左右されないようにするため)
    heights = sorted(height for _order, _text, height, _confidence in candidates if height > 0)
    median_height = heights[len(heights) // 2] if heights else 0.0
    last_position = max(len(candidates) - 1, 1)

    def score(item: tuple[int, str, float, float | None]) -> float:
        position, _text, height, confidence = item
        # 基本は読み順。画面の上にある行ほど高い点数になる
        value = (1.0 - min(position / last_position, 1.0)) * _ORDER_WEIGHT
        if median_height > 0 and height >= median_height * _HEADING_HEIGHT_RATIO:
            value += _HEADING_BONUS
        if confidence is not None:
            # 0.9を基準に、確信度が低いほど減点する
            value += (confidence - 0.9) * 3.0
        return value

    # 3. 上位を選び、元の並び順に戻す
    #    同点のときは元の順序を保つため、第2キーに元の位置を使う
    ranked = sorted(candidates, key=lambda item: (-score(item), item[0]))[:limit]
    selected = sorted(ranked, key=lambda item: item[0])
    return [
        text
        if len(text) <= _IMPORTANT_LINE_MAX_LENGTH
        else f"{text[: _IMPORTANT_LINE_MAX_LENGTH - 3]}..."
        for _order, text, _height, _confidence in selected
    ]


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
    def __init__(self, message: str, *, pack_id: str | None = None):
        super().__init__(message)
        self.pack_id = pack_id


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
            if not (
                resolved.name == pack_id
                or resolved.name.startswith(f"{pack_id}-")
                or resolved.name.startswith(f".{pack_id}-")
            ):
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
        include_appendix: bool = False,
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
                f"UPDATE screen_event SET retained=1 WHERE id IN ({placeholders})", unique_ids
            )
            connection.execute(
                """INSERT INTO context_pack(
                       id,title,query,note,deduplicate_overlaps,include_appendix,status,created_at
                   ) VALUES(?,?,?,?,?,?, 'draft', ?)""",
                (
                    pack_id,
                    document_title,
                    query.strip(),
                    note.strip(),
                    int(deduplicate_overlaps),
                    int(include_appendix),
                    created,
                ),
            )
            for position, row in enumerate(rows):
                connection.execute(
                    """INSERT INTO context_pack_item(pack_id,event_id,position,ocr_text)
                       VALUES(?,?,?,?)""",
                    (pack_id, row["id"], position, row["ocr_text"]),
                )
        try:
            return self.approve(pack_id)
        except Exception as exc:
            # The draft already exists. Expose its ID so the UI retries this
            # document rather than creating a second abandoned draft.
            raise PackError(str(exc), pack_id=pack_id) from exc

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
            result["pages"] = self.pages(result)
        return result

    def pages(self, pack: dict[str, Any]) -> list[dict[str, Any]]:
        """実際に出来上がったページの一覧を返す。

        ``items`` は「選んだ元画面」1件につき1行なので、重複をまとめても件数は減らない。
        画面に並べるべきはこちら。
        """
        manifest_path = self.artifact_dir(pack) / "manifest.json"
        if not manifest_path.exists():
            return []
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [
            {
                "position": item["position"],
                "event_id": item["event_id"],
                "started_at": item["started_at"],
                "ended_at": item["ended_at"],
                "source_event_ids": item["source_event_ids"],
                "stitched": bool(item.get("stitch", {}).get("applied")),
            }
            for item in manifest.get("items", [])
        ]

    def page_image_path(self, pack_id: str, position: int) -> Path:
        """指定ページの画像ファイルの場所を返す。"""
        pack = self.get(pack_id, include_items=False)
        directory = self.artifact_dir(pack)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise PackError("Context document has not been built yet")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        item = next(
            (value for value in manifest.get("items", []) if value["position"] == position),
            None,
        )
        if item is None:
            raise PackError("Page not found")
        # manifestの値をそのままパスに使うため、生成物の外を指していないことを必ず確かめる
        target = (directory / item["image"]).resolve()
        if not target.is_relative_to(directory.resolve()) or not target.exists():
            raise PackError("Page image is missing")
        return target

    def remove_pages(self, pack_id: str, positions: Iterable[int]) -> dict[str, Any]:
        """指定したページを文書から外して作り直す。

        外すのは文書からだけで、元の画面は画面履歴に残る(あとで別の文書に使える)。
        結合されたページは複数の元画面から出来ているため、その全部が対象になる。

        まとめて受け取るのは、作り直しが重い(実測で153枚の文書に75秒)ため。
        1枚ずつ消すと待ち時間がその回数ぶんかかる。
        """
        pack = self.get(pack_id, include_items=True)
        if pack["status"] not in {"draft", "approved"}:
            raise PackError("Only active context documents can be edited")
        wanted = list(dict.fromkeys(int(value) for value in positions))
        if not wanted:
            raise PackError("Select at least one page")
        by_position = {value["position"]: value for value in pack["pages"]}
        # 1つでも見つからなければ何も消さない。一部だけ消えて残りが黙って
        # 無視されると、利用者は何が消えたのか分からなくなる
        if any(position not in by_position for position in wanted):
            raise PackError("Page not found")
        removing = {
            event_id
            for position in wanted
            for event_id in by_position[position]["source_event_ids"]
        }
        if not any(item["event_id"] not in removing for item in pack["items"]):
            raise PackError("A context document needs at least one screen")
        # position は詰め直さない。UNIQUE(pack_id, position) があるため詰め直すと
        # 更新の途中で衝突する。読み出しは position 順なので歯抜けでも並びは保たれる
        return self._build(
            pack_id,
            removing=removing,
            expected_artifact=pack.get("artifact_path"),
        )

    def get_approved(self, pack_id: str) -> dict[str, Any]:
        pack = self.get(pack_id, include_items=True)
        if pack["status"] != "approved" or not pack["expires_at"] or pack["expires_at"] <= iso(utcnow()):
            raise PermissionError("Context pack is not approved or has expired")
        return pack

    def approve(self, pack_id: str, expiry_hours: int | None = None) -> dict[str, Any]:
        return self._build(pack_id, expiry_hours=expiry_hours)

    def configure_and_rebuild(
        self,
        pack_id: str,
        *,
        deduplicate_overlaps: bool,
        include_appendix: bool,
    ) -> dict[str, Any]:
        """Rebuild the same ordered sources without renewing or restoring sharing."""
        return self._build(
            pack_id,
            options=(deduplicate_overlaps, include_appendix),
            allow_inactive=True,
        )

    def _claim_build(
        self,
        pack_id: str,
        token: str,
        *,
        options: tuple[bool, bool] | None,
        allow_inactive: bool,
        removing: set[int] | None,
        expected_artifact: str | None,
    ) -> dict[str, Any]:
        # The claim is persisted, so two HTTP workers or ContextPackService
        # instances cannot both modify a document. Expired leases permit recovery
        # after a crashed process; completion is still fenced by the unique token.
        self.expire_due()
        now = utcnow()
        stale_before = iso(now - timedelta(minutes=15))
        with self.db.transaction() as connection:
            row = connection.execute("SELECT * FROM context_pack WHERE id=?", (pack_id,)).fetchone()
            if row is None:
                raise PackError("Context pack not found", pack_id=pack_id)
            pack = dict(row)
            if pack.get("build_token") and (pack.get("build_started_at") or "") > stale_before:
                raise PackError("Context document is already being rebuilt", pack_id=pack_id)
            if not allow_inactive and pack["status"] not in {"draft", "approved"}:
                raise PackError("Only active context documents can be rebuilt", pack_id=pack_id)
            if removing:
                if pack.get("artifact_path") != expected_artifact:
                    raise PackError("Document changed; reload its pages before editing", pack_id=pack_id)
                placeholders = ",".join("?" for _ in removing)
                remaining = connection.execute(
                    f"""SELECT COUNT(*) FROM context_pack_item
                        WHERE pack_id=? AND event_id NOT IN ({placeholders})""",
                    [pack_id, *removing],
                ).fetchone()[0]
                if not remaining:
                    raise PackError("A context document needs at least one screen", pack_id=pack_id)
                connection.execute(
                    f"DELETE FROM context_pack_item WHERE pack_id=? AND event_id IN ({placeholders})",
                    [pack_id, *removing],
                )
            if options is not None:
                pack["deduplicate_overlaps"], pack["include_appendix"] = map(int, options)
            connection.execute(
                """UPDATE context_pack SET build_token=?, build_started_at=?, build_error=NULL,
                       status=CASE WHEN status='approved' THEN 'draft' ELSE status END,
                       deduplicate_overlaps=?, include_appendix=? WHERE id=?""",
                (token, iso(now), pack["deduplicate_overlaps"], pack["include_appendix"], pack_id),
            )
            pack["items"] = [dict(item) for item in connection.execute(
                """SELECT i.*, e.started_at, e.ended_at, e.frame_path, e.thumbnail_path,
                          e.session_id, e.width, e.height, e.event_kind,
                          e.ocr_text AS source_ocr_text
                   FROM context_pack_item i JOIN screen_event e ON e.id=i.event_id
                   WHERE i.pack_id=? ORDER BY i.position""", (pack_id,)
            ).fetchall()]
        return pack

    def _build(
        self,
        pack_id: str,
        *,
        expiry_hours: int | None = None,
        options: tuple[bool, bool] | None = None,
        allow_inactive: bool = False,
        removing: set[int] | None = None,
        expected_artifact: str | None = None,
    ) -> dict[str, Any]:
        hours = expiry_hours or self.settings.default_approval_hours
        if hours < 1 or hours > 24 * 365:
            raise PackError("Approval duration must be between 1 hour and 365 days")
        token = uuid.uuid4().hex
        pack = self._claim_build(
            pack_id, token, options=options, allow_inactive=allow_inactive,
            removing=removing, expected_artifact=expected_artifact,
        )
        previous_dir = self.artifact_dir(pack)
        build_id = token[:12]
        pack_dir = self.settings.packs_dir / f".{pack_id}-{build_id}.tmp"
        final_dir = self.settings.packs_dir / f"{pack_id}-{build_id}"
        images_dir = pack_dir / "images"
        try:
            images_dir.mkdir(parents=True)
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
            # 重要行の選定が文字の大きさと確信度を使うため、テキストではなく行データで受け取る
            cleaned_line_groups = _clean_ocr_line_groups(
                [page.lines for page in rendered_pages],
                [(page.image.width, page.image.height) for page in rendered_pages],
            )
            manifest_items: list[dict[str, Any]] = []
            for position, (page, cleaned_lines) in enumerate(
                zip(rendered_pages, cleaned_line_groups, strict=True)
            ):
                ocr_text = "\n".join(line["text"] for line in cleaned_lines)
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
                        "important_ocr": _important_ocr(cleaned_lines),
                    }
                )

            if (
                pack.get("approved_at")
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
                "include_appendix": bool(pack.get("include_appendix", 0)),
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
                    current = connection.execute(
                        "SELECT build_token FROM context_pack WHERE id=?", (pack_id,)
                    ).fetchone()
                    if current is None or current["build_token"] != token:
                        raise PackError("Document rebuild was cancelled or superseded", pack_id=pack_id)
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
                        """UPDATE context_pack SET status=?, approved_at=?, expires_at=?,
                                   manifest_sha256=?, artifact_path=?, page_count=?,
                                   build_error=NULL, build_token=NULL, build_started_at=NULL
                           WHERE id=? AND build_token=?""",
                        (
                            pack["status"] if pack["status"] in {"revoked", "expired"} else "approved",
                            iso(approved_at),
                            iso(expires_at),
                            manifest_sha,
                            str(final_dir),
                            len(manifest_items),
                            pack_id,
                            token,
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
            with self.db.transaction() as connection:
                cursor = connection.execute(
                    """UPDATE context_pack SET status=?, build_error=?,
                              build_token=NULL, build_started_at=NULL WHERE id=? AND build_token=?""",
                    (
                        pack["status"] if pack["status"] in {"revoked", "expired"} else "draft",
                        str(exc)[:2000], pack_id, token,
                    ),
                )
                owned_failure = bool(cursor.rowcount)
            if not owned_failure:
                raise PackError("Document rebuild was cancelled or superseded", pack_id=pack_id) from exc
            raise

    def revoke(self, pack_id: str) -> dict[str, Any]:
        self.expire_due()
        with self.db.transaction() as connection:
            pack = connection.execute("SELECT * FROM context_pack WHERE id=?", (pack_id,)).fetchone()
            if pack is None:
                raise PackError("Context pack not found", pack_id=pack_id)
            if pack["status"] not in ("approved", "expired", "revoked") and not pack["build_token"]:
                raise PackError("Only approved or expired packs can be revoked", pack_id=pack_id)
            connection.execute(
                """UPDATE context_pack SET status='revoked', revoked_at=?,
                          build_token=NULL, build_started_at=NULL WHERE id=?""",
                (iso(utcnow()), pack_id),
            )
        return self.get(pack_id, include_items=True)

    def _artifact_directories(self, pack_id: str) -> list[Path]:
        # UUID names are produced by create(). Use exact separators rather than
        # a loose prefix so deleting one document cannot consume another's files.
        return [
            path for path in self.settings.packs_dir.iterdir()
            if path.is_dir() and not path.is_symlink() and (
                path.name == pack_id
                or path.name.startswith(f"{pack_id}-")
                or path.name.startswith(f".{pack_id}-")
            )
        ]

    @staticmethod
    def _directory_bytes(directories: list[Path]) -> int:
        total = 0
        for directory in directories:
            for path in directory.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except OSError:
                    # An in-flight build may have been cancelled or moved.
                    continue
        return total

    def deletion_info(self, pack_id: str) -> dict[str, Any]:
        pack = self.get(pack_id, include_items=False)
        counts = self.db.fetchone(
            """SELECT COUNT(*) AS source_count,
                      COALESCE(SUM(EXISTS(
                          SELECT 1 FROM context_pack_item other
                          WHERE other.event_id=i.event_id AND other.pack_id<>i.pack_id
                      )), 0) AS shared_source_count
               FROM context_pack_item i WHERE i.pack_id=?""", (pack_id,)
        )
        return {
            "pack_id": pack_id,
            "title": pack["title"],
            "document_bytes": self._directory_bytes(self._artifact_directories(pack_id)),
            "source_count": int(counts["source_count"]),
            "shared_source_count": int(counts["shared_source_count"]),
        }

    def delete(self, pack_id: str) -> dict[str, Any]:
        """Explicitly remove this document, retaining all original screen events."""
        info = self.deletion_info(pack_id)
        with self.db.transaction() as connection:
            # Keep the user's long-term originals even after their final document
            # reference disappears. History deletion is a separate explicit action.
            connection.execute(
                """UPDATE screen_event SET retained=1 WHERE id IN (
                       SELECT event_id FROM context_pack_item WHERE pack_id=?)""", (pack_id,)
            )
            cursor = connection.execute("DELETE FROM context_pack WHERE id=?", (pack_id,))
            if not cursor.rowcount:
                raise PackError("Context pack not found", pack_id=pack_id)
        # Deletion of the DB row invalidates any in-flight build token. Its late
        # completion cannot republish, and its failure cannot update another row.
        directories = self._artifact_directories(pack_id)
        before = self._directory_bytes(directories)
        for directory in directories:
            self._remove_artifact_dir(directory, pack_id)
        remaining = self._directory_bytes(self._artifact_directories(pack_id))
        return {
            **info,
            "deleted": True,
            "removed_document_bytes": max(0, before - remaining),
            "remaining_document_bytes": remaining,
        }

    def expire_due(self) -> int:
        now = iso(utcnow())
        with closing(self.db.connect()) as connection:
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
            temporary = pack_dir / f".context-{uuid.uuid4().hex}.pdf"
            try:
                self._render_pdf(html_path, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    def get_approved_document(self, pack_id: str, document_format: str = "pdf") -> Path:
        self.get_approved(pack_id)
        path = self.export_document(pack_id, document_format)
        # PDF rendering may take minutes. Respect a stop or expiry that occurred
        # while Edge was preparing the file, before handing it to the MCP caller.
        current = self.get_approved(pack_id)
        if path.parent != self.artifact_dir(current):
            raise PackError("Document changed while preparing the export", pack_id=pack_id)
        return path

    @staticmethod
    def _single_file_html(manifest: dict[str, Any], pack_dir: Path, manifest_sha: str) -> str:
        sections: list[str] = []
        appendix_sections: list[str] = []
        # 付録(OCR全文)は画面ページと同じ情報をもう一度並べるもので、枚数が増えると
        # 文書の半分以上を占める(実測で105画面のとき236ページ中129ページ)。既定では付けない
        include_appendix = bool(manifest.get("include_appendix", False))
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
            if include_appendix:
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
    /* PDF化にはEdgeの--print-to-pdfを使う。游ゴシック系・メイリオは「一」と康熙部首の
       「⼀」を同じグリフで持つため、ChromiumがToUnicode(グリフ→文字の逆引き表)を作る際に
       康熙部首側のコードポイントを採用してしまい、PDFのテキスト層が壊れる。
       (見た目は正常だが、PDF内検索・コピー・MCP経由でAIが読むテキストが別文字になる)
       BIZ UDPGothic(Windows 10 1809以降に標準搭載)とMS Gothicは正しい文字に戻るため、
       この2書体を優先する。フォントを変える際は必ずPDFからのテキスト再抽出で検証すること。 */
    :root {{
      color-scheme: dark;
      --doc-font: "BIZ UDPGothic", "MS Gothic", sans-serif;
      font-family: var(--doc-font);
    }}
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
      font: 12px/1.45 var(--doc-font); color: #d8e0e2;
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
  {f'''<section class="appendix">
    <h2>抽出テキスト付録</h2>
    <p class="meta">画面上で反復するメニューや時刻などを除外したOCR全文です。</p>
    {''.join(appendix_sections)}
  </section>''' if include_appendix else ''}
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
        # Chromium creates deep cache paths inside its profile; nesting the
        # profile below a user document can exceed Windows path limits.
        profile = Path(tempfile.mkdtemp(prefix="vm-pdf-"))
        rendered = profile / "context.pdf"
        command = [
            str(edge),
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={rendered}",
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
            while time.monotonic() < deadline and not rendered.exists():
                time.sleep(0.2)
            if process.returncode or not rendered.exists() or rendered.stat().st_size == 0:
                detail = stderr.decode(errors="replace")[-1000:]
                raise PackError(f"Unable to create PDF document: {detail or process.returncode}")
            shutil.copyfile(rendered, target)
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
