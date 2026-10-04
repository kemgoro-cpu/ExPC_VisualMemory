from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .reading_order import polygon_box, sort_in_reading_order
from .textnorm import (
    OUTER_CHROME_BOTTOM_RATIO,
    OUTER_CHROME_TOP_RATIO,
    line_key,
)

LINE_SIMILARITY = 0.96


@dataclass(slots=True)
class SourceFrame:
    event_id: int
    session_id: str
    started_at: str
    ended_at: str
    image: Image.Image
    lines: list[dict[str, Any]]
    redactions: list[dict[str, float]]


@dataclass(slots=True)
class RenderedPage:
    image: Image.Image
    lines: list[dict[str, Any]]
    source_event_ids: list[int]
    event_id: int
    started_at: str
    ended_at: str
    stitch: dict[str, Any]


@dataclass(slots=True)
class _Overlap:
    delta_y: int
    content_top: int
    content_bottom: int
    visual_score: float
    ocr_score: float
    matched_lines: int


@dataclass(slots=True)
class _PlacedFrame:
    source: SourceFrame
    offset_y: int


def _normalize_line(value: str) -> str:
    return " ".join(str(value).replace("\u3000", " ").split()).casefold()


def _line_geometry(line: dict[str, Any]) -> tuple[float, float, float] | None:
    polygon = line.get("polygon") or []
    xs = [float(point[0]) for point in polygon if len(point) >= 2]
    ys = [float(point[1]) for point in polygon if len(point) >= 2]
    if not xs or not ys:
        return None
    return min(ys), max(ys), (min(ys) + max(ys)) / 2


def _prepared_lines(source: SourceFrame, top: int, bottom: int) -> list[dict[str, Any]] | None:
    height = source.image.height
    prepared: list[dict[str, Any]] = []
    for line in source.lines:
        text = _normalize_line(line.get("text", ""))
        if not text:
            continue
        geometry = _line_geometry(line)
        if geometry is None:
            return None
        line_top, line_bottom, center = geometry
        if center < top or center >= height - bottom:
            continue
        prepared.append(
            {
                "line": line,
                "text": text,
                "top": line_top,
                "bottom": line_bottom,
                "center": center,
            }
        )
    prepared.sort(key=lambda item: (item["center"], item["text"]))
    return prepared


def _static_edge_bands(left: Image.Image, right: Image.Image) -> tuple[int, int]:
    left_gray = cv2.cvtColor(np.asarray(left.convert("RGB")), cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(np.asarray(right.convert("RGB")), cv2.COLOR_RGB2GRAY)
    difference = np.abs(left_gray.astype(np.int16) - right_gray.astype(np.int16))
    static_rows = np.mean(difference <= 2, axis=1) >= 0.995
    height = left_gray.shape[0]
    limit = max(0, int(height * 0.2))

    top = 0
    while top < limit and static_rows[top]:
        top += 1
    bottom = 0
    while bottom < limit and static_rows[height - bottom - 1]:
        bottom += 1
    return top, bottom


def _visual_correlation(
    left: Image.Image,
    right: Image.Image,
    delta_y: int,
    top: int,
    bottom: int,
) -> float:
    left_gray = cv2.cvtColor(np.asarray(left.convert("RGB")), cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(np.asarray(right.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height = left_gray.shape[0]
    left_gray = left_gray[top : height - bottom if bottom else height]
    right_gray = right_gray[top : height - bottom if bottom else height]
    content_height = left_gray.shape[0]
    if delta_y >= 0:
        left_overlap = left_gray[delta_y:]
        right_overlap = right_gray[: content_height - delta_y]
    else:
        left_overlap = left_gray[: content_height + delta_y]
        right_overlap = right_gray[-delta_y:]
    if left_overlap.shape != right_overlap.shape or min(left_overlap.shape) < 8:
        return 0.0
    if left_overlap.shape[1] > 640:
        scale = 640 / left_overlap.shape[1]
        size = (640, max(8, round(left_overlap.shape[0] * scale)))
        left_overlap = cv2.resize(left_overlap, size, interpolation=cv2.INTER_AREA)
        right_overlap = cv2.resize(right_overlap, size, interpolation=cv2.INTER_AREA)
    mean_error = float(
        np.mean(np.abs(left_overlap.astype(np.float32) - right_overlap.astype(np.float32)))
    )
    if mean_error <= 0.5:
        return 1.0
    score = float(cv2.matchTemplate(left_overlap, right_overlap, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score if np.isfinite(score) else 0.0


# 文字による裏付けが無いまま「同じ画面」と判断するときに求める視覚相関。
# 行が一致している場合の要求値(0.98)より厳しくしてある
_IDENTICAL_FRAME_CORRELATION = 0.995

# --- 視覚相関に求める水準 ---
#
# 画面には、スクロールしても動かないUI(タスクバー、アプリのリボン、左のフォルダ一覧)が
# 大きな面積を占める。少しだけスクロールした2枚を縦にずらして重ねると、この動かない部分が
# 必ずずれるため、本文がぴったり重なっていても相関は大きく下がる。
# 実測では、6〜8行が1〜2pxの誤差で揃っている明らかなスクロールでも相関は0.19〜0.36だった。
#
# そこで「文字による裏付けがどれだけ強いか」で要求水準を変える。
# 何行も連続して一致し、かつずれ量がほぼ完全に揃っているなら、それ自体が十分な証拠であり、
# 相関は「まったく別物ではない」ことの確認に留める。

# これだけの行数・文字数が連続一致し、ずれのばらつきも小さければ「強い一致」とみなす
_STRONG_MATCH_LINES = 6
_STRONG_MATCH_CHARS = 100
_STRONG_MATCH_SPREAD = 3.0

# 強い一致があるときの視覚相関の下限(別物でないことの確認)
_STRONG_MATCH_VISUAL = 0.10


def _required_visual_score(
    delta_y: int, length: int, character_count: int, spread: float
) -> float:
    """この一致を採用するために必要な視覚相関を返す。"""
    if delta_y == 0:
        # ずらさずに重なる=同じ画面。動かないUIも含めて一致するはずなので厳しく見る
        return 0.98
    if (
        length >= _STRONG_MATCH_LINES
        and character_count >= _STRONG_MATCH_CHARS
        and spread <= _STRONG_MATCH_SPREAD
    ):
        return _STRONG_MATCH_VISUAL
    return 0.92


def _identical_frame_overlap(
    left: SourceFrame,
    right: SourceFrame,
    top: int,
    bottom: int,
) -> _Overlap | None:
    """文字の一致を使わず、画像が同一かどうかだけで重複を判定する。

    マウスカーソルの移動やホバー効果だけが違うフレームは、ずらさずに(delta_y=0で)
    重なる。文字が3行以上取れる画面なら通常の行の照合で拾えるが、文字がほとんど
    無い画面はそこへ到達できないため、ここで受け止める。
    """
    score = _visual_correlation(left.image, right.image, 0, top, bottom)
    if score < _IDENTICAL_FRAME_CORRELATION or not _same_frame_pixels(left.image, right.image, top, bottom):
        return None
    return _Overlap(
        delta_y=0,
        content_top=top,
        content_bottom=bottom,
        visual_score=score,
        ocr_score=0.0,
        matched_lines=0,
    )


def _find_overlap(
    left: SourceFrame,
    right: SourceFrame,
    forced_bands: tuple[int, int] | None = None,
) -> _Overlap | None:
    if left.image.size != right.image.size:
        return None
    top, bottom = forced_bands or _static_edge_bands(left.image, right.image)
    geometries = [
        geometry
        for source in (left, right)
        for line in source.lines
        if (geometry := _line_geometry(line)) is not None
    ]
    if geometries:
        top = min(top, max(0, int(min(geometry[0] for geometry in geometries))))
        bottom = min(
            bottom,
            max(0, left.image.height - int(max(geometry[1] for geometry in geometries))),
        )
    content_height = left.image.height - top - bottom
    if content_height < 16:
        return None
    left_lines = _prepared_lines(left, top, bottom)
    right_lines = _prepared_lines(right, top, bottom)
    if left_lines is None or right_lines is None:
        # Without positions, group rendering cannot place the OCR onto the
        # stitched page. Leave these screens to the final duplicate pass, which
        # preserves their complete lines instead of dropping unknown geometry.
        return None
    if len(left_lines) < 3 or len(right_lines) < 3:
        # 文字がほとんど取れない画面(画像主体のページ、読み込み中の画面など)は
        # 行の一致で判断できない。画像そのものが同じなら重複として扱う
        return _identical_frame_overlap(left, right, top, bottom)

    rows, columns = len(left_lines), len(right_lines)
    runs = np.zeros((rows, columns), dtype=np.int16)
    similarities = np.zeros((rows, columns), dtype=np.float32)
    candidates: list[tuple[int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            similarity = SequenceMatcher(
                None, left_lines[row]["text"], right_lines[column]["text"]
            ).ratio()
            similarities[row, column] = similarity
            if similarity < LINE_SIMILARITY:
                continue
            runs[row, column] = 1 + (runs[row - 1, column - 1] if row and column else 0)
            if runs[row, column] >= 3:
                candidates.append((int(runs[row, column]), row, column))

    best: _Overlap | None = None
    best_rank: tuple[int, int, float] | None = None
    for length, row, column in candidates:
        left_run = left_lines[row - length + 1 : row + 1]
        right_run = right_lines[column - length + 1 : column + 1]
        character_count = sum(
            min(len(a["text"]), len(b["text"]))
            for a, b in zip(left_run, right_run, strict=True)
        )
        if character_count < 30:
            continue
        deltas = np.asarray(
            [
                a["center"] - b["center"]
                for a, b in zip(left_run, right_run, strict=True)
            ],
            dtype=np.float32,
        )
        delta_y = int(round(float(np.median(deltas))))
        spread = float(np.max(np.abs(deltas - delta_y)))
        if spread > max(4.0, left.image.height * 0.015):
            continue
        overlap_height = content_height - abs(delta_y)
        if overlap_height < max(8, content_height * 0.08):
            continue
        visual_score = _visual_correlation(left.image, right.image, delta_y, top, bottom)
        required_visual = _required_visual_score(delta_y, length, character_count, spread)
        if visual_score < required_visual:
            continue
        if delta_y == 0 and not _same_frame_pixels(left.image, right.image, top, bottom):
            continue
        start_row = row - length + 1
        start_column = column - length + 1
        ocr_score = float(
            np.mean(
                [similarities[start_row + index, start_column + index] for index in range(length)]
            )
        )
        rank = (length, character_count, visual_score)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best = _Overlap(
                delta_y=delta_y,
                content_top=top,
                content_bottom=bottom,
                visual_score=visual_score,
                ocr_score=ocr_score,
                matched_lines=length,
            )
    if best is None:
        # 行では一致を見つけられなかった場合も、画像が同一なら重複として拾う
        return _identical_frame_overlap(left, right, top, bottom)
    return best


def _compatible(left: SourceFrame, right: SourceFrame) -> bool:
    if left.session_id != right.session_id or left.image.size != right.image.size:
        return False
    try:
        left_time = datetime.fromisoformat(left.started_at)
        right_time = datetime.fromisoformat(right.started_at)
    except ValueError:
        return False
    return abs((right_time - left_time).total_seconds()) <= 120


def _similar_at_position(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_geometry = _line_geometry(left)
    right_geometry = _line_geometry(right)
    if left_geometry is None or right_geometry is None:
        return False
    if abs(left_geometry[2] - right_geometry[2]) > max(4.0, left_geometry[1] - left_geometry[0]):
        return False
    return SequenceMatcher(
        None, _normalize_line(left.get("text", "")), _normalize_line(right.get("text", ""))
    ).ratio() >= LINE_SIMILARITY


# 結合した縦長の画像を1画面ぶんずつに切り分けるとき、最後に残る切れ端が薄いと
# 「タスクバーだけが写った高さ42pxのページ」のような、内容の無いページができてしまう。
# 実測では1行(約20px)スクロールしただけの結合で 1069px + 42px に割れていた。
# 画面の高さのこの割合に満たない切れ端は作らず、手前のページに含めて出す。
_MIN_TAIL_FRACTION = 0.25


def _page_boundaries(total_height: int, viewport_height: int, lines: list[dict[str, Any]]) -> list[int]:
    boundaries = [0]
    # 1画面をわずかに超えるだけの高さなら分割しない(切れ端を作らない)
    minimum_tail = viewport_height * _MIN_TAIL_FRACTION
    boxes = sorted(
        geometry
        for line in lines
        if (geometry := _line_geometry(line)) is not None
    )
    gaps = [
        (left[1] + right[0]) / 2
        for left, right in zip(boxes, boxes[1:], strict=False)
        if right[0] > left[1]
    ]
    start = 0
    while total_height - start > viewport_height + minimum_tail:
        target = start + viewport_height
        radius = viewport_height * 0.2
        candidates = [
            value
            for value in gaps
            if start + viewport_height * 0.5 <= value <= target and abs(value - target) <= radius
        ]
        boundary = (
            int(round(min(candidates, key=lambda value: abs(value - target))))
            if candidates
            else target
        )
        if boundary <= start:
            boundary = target
        if total_height - boundary < minimum_tail:
            # 行の切れ目に合わせた結果、最後が薄くなるなら分割をやめる
            break
        boundaries.append(boundary)
        start = boundary
    boundaries.append(total_height)
    return boundaries


def _render_single(source: SourceFrame) -> RenderedPage:
    lines = [{**line, "event_id": source.event_id} for line in source.lines]
    return RenderedPage(
        image=source.image.copy(),
        lines=lines,
        source_event_ids=[source.event_id],
        event_id=source.event_id,
        started_at=source.started_at,
        ended_at=source.ended_at,
        stitch={"applied": False, "confidence": None, "placements": []},
    )


def _render_group(
    placed: list[_PlacedFrame],
    top: int,
    bottom: int,
    overlaps: list[_Overlap],
) -> list[RenderedPage]:
    if len(placed) == 1:
        return [_render_single(placed[0].source)]
    width, height = placed[0].source.image.size
    content_height = height - top - bottom
    minimum = min(item.offset_y for item in placed)
    maximum = max(item.offset_y + content_height for item in placed)
    content_extent = maximum - minimum
    total_height = top + content_extent + bottom

    global_lines: list[dict[str, Any]] = []
    for item in placed:
        for raw_line in item.source.lines:
            geometry = _line_geometry(raw_line)
            if geometry is None or geometry[2] < top or geometry[2] >= height - bottom:
                continue
            polygon = [
                [float(point[0]), top + item.offset_y - minimum + float(point[1]) - top]
                for point in raw_line.get("polygon", [])
                if len(point) >= 2
            ]
            line = {**raw_line, "polygon": polygon, "event_id": item.source.event_id}
            if any(_similar_at_position(existing, line) for existing in global_lines):
                continue
            global_lines.append(line)
    # 結合後のページも段組みを考慮した読み順に並べ直す。
    # (縦位置だけで並べると、2段組みの左右の行が1行ずつ交互に混ざる)
    global_lines = sort_in_reading_order(
        global_lines, lambda line: polygon_box(line.get("polygon"))
    )

    boundaries = _page_boundaries(total_height, height, global_lines)
    confidence = min(
        min(overlap.visual_score, overlap.ocr_score) for overlap in overlaps
    )
    pages: list[RenderedPage] = []
    for page_index, (page_top, page_bottom) in enumerate(
        zip(boundaries, boundaries[1:], strict=False)
    ):
        page_height = page_bottom - page_top
        canvas = Image.new("RGB", (width, page_height), "black")

        header_end = top
        if top and page_top < header_end and page_bottom > 0:
            crop_top, crop_bottom = max(page_top, 0), min(page_bottom, header_end)
            crop = placed[0].source.image.crop((0, crop_top, width, crop_bottom))
            canvas.paste(crop, (0, crop_top - page_top))

        contributing: list[int] = []
        placements: list[dict[str, Any]] = []
        for item in placed:
            logical_top = top + item.offset_y - minimum
            logical_bottom = logical_top + content_height
            intersect_top = max(page_top, logical_top)
            intersect_bottom = min(page_bottom, logical_bottom)
            if intersect_top >= intersect_bottom:
                continue
            source_top = top + intersect_top - logical_top
            source_bottom = top + intersect_bottom - logical_top
            crop = item.source.image.crop((0, source_top, width, source_bottom))
            canvas.paste(crop, (0, intersect_top - page_top))
            contributing.append(item.source.event_id)
            placements.append(
                {
                    "event_id": item.source.event_id,
                    "output_y": logical_top - page_top,
                    "content_top": top,
                    "content_bottom": height - bottom,
                }
            )

        footer_start = top + content_extent
        if bottom and page_top < total_height and page_bottom > footer_start:
            intersect_top = max(page_top, footer_start)
            intersect_bottom = min(page_bottom, total_height)
            source_top = height - bottom + intersect_top - footer_start
            source_bottom = height - bottom + intersect_bottom - footer_start
            crop = placed[-1].source.image.crop((0, source_top, width, source_bottom))
            canvas.paste(crop, (0, intersect_top - page_top))
            if placed[-1].source.event_id not in contributing:
                contributing.append(placed[-1].source.event_id)

        page_lines: list[dict[str, Any]] = []
        for line in global_lines:
            geometry = _line_geometry(line)
            if geometry is None or not page_top <= geometry[2] < page_bottom:
                continue
            polygon = [
                [float(point[0]), float(point[1]) - page_top]
                for point in line.get("polygon", [])
                if len(point) >= 2
            ]
            page_lines.append({**line, "polygon": polygon})

        if not contributing:
            contributing = [placed[0].source.event_id]
        pages.append(
            RenderedPage(
                image=canvas,
                lines=page_lines,
                source_event_ids=list(dict.fromkeys(contributing)),
                event_id=contributing[0],
                started_at=placed[0].source.started_at,
                ended_at=placed[-1].source.ended_at,
                stitch={
                    "applied": True,
                    "confidence": round(confidence, 4),
                    "page_index": page_index,
                    "placements": placements,
                },
            )
        )
    return pages


# --- 離れたページ同士の重複をまとめる ---
#
# ここまでの結合は「隣り合うフレーム」しか比べていない。実際の操作では
# 資料 → 別アプリ → 同じ資料 のように行き来するため、同じ画面が離れた位置に
# 何度も現れる(実測では105ページ中21ページが再登場だった)。
#
# 文字の一致だけではグラフ・写真の差異を見落とすため、ここでは画像一致を
# 必須にする。スクロールの重なりは前段の位置合わせで検証する。
# Pixel amplitude tolerance handles small encoding noise. Do not average the
# changed area across the screen: a one-digit or narrow graph change matters.
_ABSORB_PIXEL_DELTA = 12
# 総当たりの画素比較は重いので、粗い縮小画像で明らかに違うものを先に外す。
# pHash も試したが、白紙や壁紙のように濃淡の乏しい画面では値が安定せず、
# 同じ画面どうしでも距離が開いて取りこぼした(実測で3ページぶん)。
# 縮小した画像の平均的な明るさの差なら、そうした画面でも素直に効く。
_ABSORB_COARSE_SIZE = 16
_ABSORB_COARSE_DELTA = 12.0


def _page_pixels(page: RenderedPage) -> np.ndarray:
    """ページ画像を、細いグラフや文字の差も残る元解像度で比較する。

    タスクバーやタイトルバーは常に写っているうえ、時計・通知バッジ・開いているアプリの
    アイコンが絶えず変わる。ここを含めて比べると、中身がまったく同じ画面でも
    「違う」と出てしまう(実測: 白紙のスライド同士で 0.0081、ふちを除くと 0.0008)。
    文字側で既に同じ帯をノイズとして落としているので、画像でも同じ帯を外して比べる。
    """
    pixels = np.asarray(page.image.convert("RGB"))
    height = pixels.shape[0]
    top = int(height * OUTER_CHROME_TOP_RATIO)
    bottom = int(height * OUTER_CHROME_BOTTOM_RATIO)
    if bottom - top < 8:
        return pixels
    return pixels[top:bottom]


def _same_pixels_or_moved_cursor(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape:
        return False
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    changed = delta > _ABSORB_PIXEL_DELTA
    if changed.ndim == 3:
        changed = np.any(changed, axis=2)
    if not np.any(changed):
        return True

    # Only ignore a confidently identified movement: two small, separated
    # regions exchange the same foreground/background patch. A changed number,
    # a growing bar, or an isolated content edit does not meet this condition.
    count, _labels, stats, _centers = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
    if count != 3:
        return False
    first, second = stats[1], stats[2]
    x1, y1, width1, height1, _area1 = map(int, first)
    x2, y2, width2, height2, _area2 = map(int, second)
    if (width1, height1) != (width2, height2) or max(width1, height1) > 32:
        return False
    if max(abs(x1 - x2), abs(y1 - y2)) < max(32, max(width1, height1) * 2):
        return False
    left_first = left[y1:y1 + height1, x1:x1 + width1].astype(np.int16)
    right_first = right[y1:y1 + height1, x1:x1 + width1].astype(np.int16)
    left_second = left[y2:y2 + height2, x2:x2 + width2].astype(np.int16)
    right_second = right[y2:y2 + height2, x2:x2 + width2].astype(np.int16)
    return bool(
        np.max(np.abs(left_first - right_second)) <= _ABSORB_PIXEL_DELTA
        and np.max(np.abs(right_first - left_second)) <= _ABSORB_PIXEL_DELTA
    )


def _same_frame_pixels(left: Image.Image, right: Image.Image, top: int, bottom: int) -> bool:
    """Grayscale correlation alone cannot distinguish equal-luminance chart colors."""
    left_pixels = np.asarray(left.convert("RGB"))[top : left.height - bottom]
    right_pixels = np.asarray(right.convert("RGB"))[top : right.height - bottom]
    return _same_pixels_or_moved_cursor(left_pixels, right_pixels)


@dataclass(slots=True)
class _AbsorbEntry:
    page: RenderedPage
    pixels: np.ndarray | None = None
    coarse: np.ndarray | None = None

    def small_image(self) -> np.ndarray:
        if self.pixels is None:
            self.pixels = _page_pixels(self.page)
        return self.pixels

    def coarse_image(self) -> np.ndarray:
        # ふちを除いた同じ画像から作る(全体から作ると、タスクバーの違いだけで弾かれる)
        if self.coarse is None:
            self.coarse = cv2.resize(
                self.small_image(),
                (_ABSORB_COARSE_SIZE, _ABSORB_COARSE_SIZE),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
        return self.coarse


def _same_content(new: _AbsorbEntry, earlier: _AbsorbEntry) -> bool:
    """new が earlier に何も足していないか。"""
    # 同じ凡例・見出しでも、グラフや写真だけが変わる画面がある。
    # OCRの包含関係だけでは画像を捨てない。正しく確認できるスクロールは
    # 前段の位置合わせで結合し、この最終吸収では画像一致を必須にする。
    if new.page.image.size != earlier.page.image.size:
        return False
    coarse_delta = float(np.mean(np.abs(new.coarse_image() - earlier.coarse_image())))
    if coarse_delta > _ABSORB_COARSE_DELTA:
        return False
    return _same_pixels_or_moved_cursor(earlier.small_image(), new.small_image())


def _absorb_repeated_pages(pages: list[RenderedPage]) -> list[RenderedPage]:
    """前に出たページと同じ内容のページを、そのページにまとめる。"""
    kept: list[_AbsorbEntry] = []

    for page in pages:
        entry = _AbsorbEntry(page=page)
        target = next(
            (
                earlier
                for earlier in reversed(kept)
                if _same_content(entry, earlier)
            ),
            None,
        )
        if target is None:
            kept.append(entry)
            continue
        # 消したページの根拠イベントと、まだ出ていない行は残す側へ引き継ぐ。
        # 見た目で畳んだ場合でも、これで文字が失われることはない
        target.page.source_event_ids = list(
            dict.fromkeys(target.page.source_event_ids + page.source_event_ids)
        )
        known = {line_key(str(line.get("text", ""))) for line in target.page.lines}
        target.page.lines = target.page.lines + [
            line
            for line in page.lines
            if line_key(str(line.get("text", ""))) not in known
        ]
    return [entry.page for entry in kept]


def render_document_pages(
    sources: list[SourceFrame], deduplicate_overlaps: bool
) -> list[RenderedPage]:
    if not deduplicate_overlaps:
        return [_render_single(source) for source in sources]

    pages: list[RenderedPage] = []
    placed = [_PlacedFrame(sources[0], 0)] if sources else []
    overlaps: list[_Overlap] = []
    bands: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal placed, overlaps, bands
        if not placed:
            return
        top, bottom = bands or (0, 0)
        pages.extend(_render_group(placed, top, bottom, overlaps))
        placed = []
        overlaps = []
        bands = None

    for source in sources[1:]:
        previous = placed[-1].source
        overlap = _find_overlap(previous, source, bands) if _compatible(previous, source) else None
        if overlap is None:
            flush()
            placed = [_PlacedFrame(source, 0)]
            continue
        if bands is None:
            bands = (overlap.content_top, overlap.content_bottom)
        placed.append(_PlacedFrame(source, placed[-1].offset_y + overlap.delta_y))
        overlaps.append(overlap)
    flush()
    return _absorb_repeated_pages(pages)
