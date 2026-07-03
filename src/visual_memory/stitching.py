from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

import cv2
import numpy as np
from PIL import Image

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
    if left_lines is None or right_lines is None or len(left_lines) < 3 or len(right_lines) < 3:
        return None

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
        if float(np.max(np.abs(deltas - delta_y))) > max(4.0, left.image.height * 0.015):
            continue
        overlap_height = content_height - abs(delta_y)
        if overlap_height < max(8, content_height * 0.08):
            continue
        visual_score = _visual_correlation(left.image, right.image, delta_y, top, bottom)
        required_visual = 0.98 if delta_y == 0 else 0.92
        if visual_score < required_visual:
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


def _page_boundaries(total_height: int, viewport_height: int, lines: list[dict[str, Any]]) -> list[int]:
    boundaries = [0]
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
    while total_height - start > viewport_height:
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
    global_lines.sort(key=lambda line: (_line_geometry(line) or (0.0, 0.0, 0.0))[2])

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
    return pages
