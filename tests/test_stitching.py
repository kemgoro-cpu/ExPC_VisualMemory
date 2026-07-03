from __future__ import annotations

from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
from PIL import Image

from visual_memory.stitching import SourceFrame, render_document_pages

WIDTH = 320
HEIGHT = 180
CONTENT_TOP = 20
CONTENT_BOTTOM = 20
LINE_STEP = 18
VISIBLE_LINES = 7


def _frame(start: int, *, visual_noise: bool = False) -> tuple[Image.Image, list[dict]]:
    image = np.full((HEIGHT, WIDTH, 3), 8, dtype=np.uint8)
    image[:CONTENT_TOP] = (35, 35, 35)
    image[HEIGHT - CONTENT_BOTTOM :] = (55, 55, 55)
    lines = []
    for position in range(VISIBLE_LINES):
        line_number = start + position
        text = f"result_{line_number:02d} = transform(source_{line_number:02d}, retry={line_number + 1})"
        baseline = CONTENT_TOP + 16 + position * LINE_STEP
        cv2.putText(
            image,
            text,
            (8, baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (220, 220, 220),
            1,
            cv2.LINE_8,
        )
        lines.append(
            {
                "text": text,
                "confidence": 0.99,
                "polygon": [
                    [6, baseline - 12],
                    [WIDTH - 6, baseline - 12],
                    [WIDTH - 6, baseline + 3],
                    [6, baseline + 3],
                ],
            }
        )
    if visual_noise:
        image[CONTENT_TOP : HEIGHT - CONTENT_BOTTOM] = np.random.default_rng(4).integers(
            0, 256, (HEIGHT - CONTENT_TOP - CONTENT_BOTTOM, WIDTH, 3), dtype=np.uint8
        )
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), lines


def _source(
    event_id: int,
    start_line: int,
    seconds: int,
    *,
    session_id: str = "recording",
    visual_noise: bool = False,
    polygons: bool = True,
) -> SourceFrame:
    image, lines = _frame(start_line, visual_noise=visual_noise)
    if not polygons:
        lines = [{**line, "polygon": []} for line in lines]
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return SourceFrame(
        event_id=event_id,
        session_id=session_id,
        started_at=stamp.isoformat(),
        ended_at=(stamp + timedelta(seconds=1)).isoformat(),
        image=image,
        lines=lines,
        redactions=[],
    )


def _all_text(pages) -> list[str]:
    return [line["text"] for page in pages for line in page.lines]


def test_downward_scroll_stitches_once_and_splits_at_viewport_height():
    first = _source(1, 0, 0)
    second = _source(2, 3, 10)

    pages = render_document_pages([first, second], True)

    assert len(pages) == 2
    assert all(page.image.height <= HEIGHT for page in pages)
    assert all(page.stitch["applied"] for page in pages)
    assert {event_id for page in pages for event_id in page.source_event_ids} == {1, 2}
    text = _all_text(pages)
    assert len(text) == 10
    assert len(set(text)) == 10
    assert text[0].startswith("result_00")
    assert text[-1].startswith("result_09")
    assert sum(page.image.height for page in pages) == HEIGHT + 3 * LINE_STEP


def test_upward_scroll_stitches_in_document_order_without_duplicate_lines():
    later = _source(2, 3, 10)
    earlier = _source(1, 0, 0)

    pages = render_document_pages([later, earlier], True)

    text = _all_text(pages)
    assert len(text) == 10
    assert len(set(text)) == 10
    assert text[0].startswith("result_00")
    assert text[-1].startswith("result_09")
    assert all(page.stitch["applied"] for page in pages)


def test_complete_duplicate_is_absorbed_into_one_page():
    first = _source(1, 0, 0)
    duplicate = _source(2, 0, 10)

    pages = render_document_pages([first, duplicate], True)

    assert len(pages) == 1
    assert pages[0].source_event_ids == [1, 2]
    assert len(_all_text(pages)) == VISIBLE_LINES


def test_low_confidence_missing_geometry_and_incompatible_frames_are_preserved():
    first = _source(1, 0, 0)
    noisy = _source(2, 3, 10, visual_noise=True)
    missing_geometry = _source(3, 3, 20, polygons=False)
    other_session = _source(4, 3, 30, session_id="other")
    late = _source(5, 3, 300)

    for right in (noisy, missing_geometry, other_session, late):
        pages = render_document_pages([first, right], True)
        assert len(pages) == 2
        assert not any(page.stitch["applied"] for page in pages)


def test_disabled_overlap_removal_preserves_original_frames_and_ocr():
    first = _source(1, 0, 0)
    second = _source(2, 3, 10)

    pages = render_document_pages([first, second], False)

    assert len(pages) == 2
    assert all(page.image.size == (WIDTH, HEIGHT) for page in pages)
    assert len(_all_text(pages)) == VISIBLE_LINES * 2
    assert not any(page.stitch["applied"] for page in pages)
