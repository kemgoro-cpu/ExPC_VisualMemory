from __future__ import annotations

from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

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


# --- 文字が取れない画面の重複判定 ---
#
# マウスカーソルが動いただけの画面は、文字が3行以上取れれば通常の行の照合で結合できる。
# 文字がほとんど無い画面(画像主体のページ、読み込み中の画面)はそこへ到達できないため、
# 画像そのものの一致で受け止める。


def _textless_source(
    event_id: int,
    seconds: int,
    *,
    cursor_at: tuple[int, int] | None = None,
    highlight: bool = False,
):
    """文字を持たない画面。

    cursor_at はその位置に小さなカーソルを描く(画面の中身は同じ扱い)。
    highlight は画面の中身そのものが変わった状態を作る。
    """
    image = np.full((HEIGHT, WIDTH, 3), 8, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (280, 140), (70, 70, 70), -1)
    if highlight:
        cv2.rectangle(image, (60, 60), (260, 120), (200, 200, 200), -1)
    if cursor_at is not None:
        x, y = cursor_at
        cv2.rectangle(image, (x, y), (x + 4, y + 7), (255, 255, 255), -1)
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return SourceFrame(
        event_id=event_id,
        session_id="recording",
        started_at=stamp.isoformat(),
        ended_at=(stamp + timedelta(seconds=1)).isoformat(),
        image=Image.fromarray(image),
        lines=[],
        redactions=[],
    )


def test_textless_frames_that_differ_only_by_the_cursor_are_merged():
    pages = render_document_pages(
        [
            _textless_source(1, 0, cursor_at=(100, 100)),
            _textless_source(2, 3, cursor_at=(160, 60)),
        ],
        True,
    )
    assert len(pages) == 1
    assert pages[0].source_event_ids == [1, 2]


def test_textless_frames_with_different_content_are_kept_apart():
    # 画面の中身がはっきり変わった場合は別ページのまま
    pages = render_document_pages(
        [_textless_source(1, 0), _textless_source(2, 3, highlight=True)], True
    )
    assert len(pages) == 2


# --- 動かないUIが大きい画面のスクロール ---
#
# タスクバーやアプリのリボン、左のフォルダ一覧はスクロールしても動かない。
# これらが画面の大半を占めると、本文がぴったり重なっていても縦にずらした比較では
# 相関が大きく下がる(実測で0.19〜0.36)。文字の一致が十分強いときは相関の要求を下げる。


def _frame_with_fixed_ui(start_line: int, *, ui_ratio: float = 0.55):
    """画面の左側 ui_ratio ぶんが「動かないUI」で、右側だけがスクロールする画面。"""
    image = np.full((HEIGHT, WIDTH, 3), 8, dtype=np.uint8)
    split = int(WIDTH * ui_ratio)
    # 動かないUI: どのフレームでも同じ絵柄
    for row in range(0, HEIGHT, 12):
        cv2.rectangle(image, (6, row + 2), (split - 10, row + 9), (60, 60, 60), -1)
    lines = []
    for index in range(VISIBLE_LINES):
        y = CONTENT_TOP + index * LINE_STEP
        # 実際の本文に近い長さにする(緩和は「6行以上かつ100文字以上」の一致が条件のため)
        text = f"本文の{start_line + index}行目：ここには説明の文章が入っている"
        cv2.putText(image, f"L{start_line + index}", (split + 10, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1)
        lines.append({
            "text": text,
            "polygon": [[split + 10, y], [WIDTH - 6, y], [WIDTH - 6, y + 14], [split + 10, y + 14]],
        })
    return Image.fromarray(image), lines


def _ui_source(event_id: int, start_line: int, seconds: int) -> SourceFrame:
    image, lines = _frame_with_fixed_ui(start_line)
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return SourceFrame(
        event_id=event_id,
        session_id="recording",
        started_at=stamp.isoformat(),
        ended_at=(stamp + timedelta(seconds=1)).isoformat(),
        image=image,
        lines=lines,
        redactions=[],
    )


def test_scroll_is_merged_even_when_fixed_ui_lowers_the_visual_score():
    # 1行ぶんスクロール = 7行中6行が重なる
    pages = render_document_pages([_ui_source(1, 0, 0), _ui_source(2, 1, 4)], True)

    assert all(page.stitch["applied"] for page in pages)
    texts = _all_text(pages)
    # 重なった6行は1度ずつになり、両端の行も残る(結合で内容を落とさない・増やさない)
    assert len(texts) == len(set(texts)) == VISIBLE_LINES + 1
    assert texts[0].startswith("本文の0行目")
    assert texts[-1].startswith("本文の7行目")


def test_unrelated_screens_are_not_merged_by_the_relaxed_rule():
    # 本文がまったく違えば、動かないUIが同じでも結合しない
    first = _ui_source(1, 0, 0)
    second = _ui_source(2, 500, 4)
    assert len(render_document_pages([first, second], True)) == 2


# --- 切れ端ページを作らない ---
#
# 1行だけスクロールした2枚を結合すると、画面の高さをわずかに超える縦長の画像になる。
# これを機械的に1画面ぶんずつ切ると、最後にタスクバーだけが写った薄いページが残る
# (実データで 1069px + 42px に割れていた)。


def test_slightly_taller_than_viewport_stays_one_page():
    pages = render_document_pages([_source(1, 0, 0), _source(2, 1, 4)], True)

    assert len(pages) == 1
    assert pages[0].image.height == HEIGHT + LINE_STEP
    text = _all_text(pages)
    assert len(text) == len(set(text)) == VISIBLE_LINES + 1


# --- 離れたページ同士の重複をまとめる ---
#
# 資料 → 別アプリ → 同じ資料 と行き来すると、同じ画面が離れた位置に何度も現れる。
# 隣り合うフレームしか見ない結合では拾えないため、ページを組んだあとにまとめる。


def _garbled_source(event_id: int, seconds: int, texts: list[str]) -> SourceFrame:
    """画像は同じで、OCRの読み取り結果だけが違うフレーム。

    スライド一覧のような細かい画面では、同じ絵でもOCRがまったく違う文字を返す。
    """
    image, _ = _frame(0)
    lines = [
        {
            "text": text,
            "confidence": 0.4,
            "polygon": [
                [6, CONTENT_TOP + index * LINE_STEP],
                [WIDTH - 6, CONTENT_TOP + index * LINE_STEP],
                [WIDTH - 6, CONTENT_TOP + index * LINE_STEP + 14],
                [6, CONTENT_TOP + index * LINE_STEP + 14],
            ],
        }
        for index, text in enumerate(texts)
    ]
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return SourceFrame(
        event_id=event_id,
        session_id="recording",
        started_at=stamp.isoformat(),
        ended_at=(stamp + timedelta(seconds=1)).isoformat(),
        image=image,
        lines=lines,
        redactions=[],
    )


def test_screen_shown_again_later_is_folded_into_its_first_appearance():
    pages = render_document_pages(
        [_source(1, 0, 0), _source(2, 100, 10), _source(3, 0, 20)], True
    )

    assert len(pages) == 2
    # 消えたページの根拠イベントは残る
    assert pages[0].source_event_ids == [1, 3]


def test_screen_that_brings_new_lines_is_kept():
    pages = render_document_pages(
        [_source(1, 0, 0), _source(2, 100, 10), _source(3, 200, 20)], True
    )

    assert len(pages) == 3


def test_same_picture_with_unreadable_text_is_folded_and_keeps_its_lines():
    # 絵は同じだがOCRの結果が食い違う場合でも、絵で同じ画面と判断する
    first = _garbled_source(1, 0, ["設計の要点まとめ", "変更点と変化点", "評価の観点", "報告の流れ"])
    other = _source(2, 100, 10)
    again = _garbled_source(3, 20, ["設計の要占まとめ", "変更占と変化占", "評伍の観占", "报告の流れ"])

    pages = render_document_pages([first, other, again], True)

    assert len(pages) == 2
    assert pages[0].source_event_ids == [1, 3]
    # 読み違えた側の行も捨てずに残す(絵で畳んでも文字は失わない)
    texts = {line["text"] for line in pages[0].lines}
    assert "設計の要点まとめ" in texts
    assert "設計の要占まとめ" in texts


# --- 画面のふちの違いは無視する ---
#
# タスクバーは常に写っていて、時計・通知バッジ・開いているアプリのアイコンが絶えず変わる。
# ここを含めて比べると、中身がまったく同じ白紙のスライドでも別ページとして残ってしまう。


def _marked_source(event_id: int, seconds: int, *, mark_y: int) -> SourceFrame:
    """指定した高さに目印を描いた、ほぼ白紙のフレーム。"""
    image = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)
    cv2.rectangle(image, (10, mark_y), (150, mark_y + 12), (20, 20, 20), -1)
    stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return SourceFrame(
        event_id=event_id,
        session_id="recording",
        started_at=stamp.isoformat(),
        ended_at=(stamp + timedelta(seconds=1)).isoformat(),
        image=Image.fromarray(image),
        lines=[],
        redactions=[],
    )


def test_difference_only_in_the_taskbar_band_is_ignored():
    # 画面の下端(タスクバーの帯)だけが違う2枚は同じ画面として畳む
    pages = render_document_pages(
        [
            _marked_source(1, 0, mark_y=HEIGHT - 14),
            _source(2, 100, 10),
            _marked_source(3, 20, mark_y=HEIGHT - 28),
        ],
        True,
    )

    assert len(pages) == 2
    assert pages[0].source_event_ids == [1, 3]


def test_difference_in_the_content_area_keeps_pages_apart():
    # 同じ位置関係でも、中身の帯が違えば別ページのまま残す
    pages = render_document_pages(
        [
            _marked_source(1, 0, mark_y=HEIGHT // 2),
            _source(2, 100, 10),
            _marked_source(3, 20, mark_y=HEIGHT // 2 + 30),
        ],
        True,
    )

    assert len(pages) == 3


def test_disabled_deduplication_keeps_every_repeated_screen():
    pages = render_document_pages(
        [_source(1, 0, 0), _source(2, 100, 10), _source(3, 0, 20)], False
    )

    assert len(pages) == 3


@pytest.mark.parametrize("color_only", [False, True])
def test_equal_ocr_does_not_discard_a_changed_graph(color_only):
    sources = []
    for index in range(2):
        image = Image.new("RGB", (800, 600), "white")
        drawing = ImageDraw.Draw(image)
        lines = []
        for offset, text in enumerate((
            "Revenue by quarter", "2023 actual result", "2024 actual result", "Units in millions",
        )):
            y = 180 + offset * 35
            drawing.text((40, y), text, fill="black")
            lines.append({"text": text, "confidence": .99,
                          "polygon": [[40, y], [240, y], [240, y + 16], [40, y + 16]]})
        if color_only:
            # Both colors round to the same grayscale luminance in OpenCV.
            drawing.line([(300, 450), (700, 160)],
                         fill=(255, 0, 0) if index else (0, 130, 0), width=20)
        else:
            drawing.line([(300, 160 if index else 450), (700, 450 if index else 160)],
                         fill="blue", width=20)
        stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index * 10)
        sources.append(SourceFrame(index + 1, "recording", stamp.isoformat(),
                                   stamp.isoformat(), image, lines, []))

    pages = render_document_pages(sources, True)

    assert len(pages) == 2
    assert [page.source_event_ids for page in pages] == [[1], [2]]
    assert np.array_equal(np.asarray(pages[0].image), np.asarray(sources[0].image))
    assert np.array_equal(np.asarray(pages[1].image), np.asarray(sources[1].image))


@pytest.mark.parametrize("change", ["number", "bar", "both"])
@pytest.mark.parametrize("with_ocr", [False, True])
def test_small_local_change_is_not_averaged_away(change, with_ocr):
    sources = []
    for index in range(2):
        image = Image.new("RGB", (960, 540), "#f1f5fa")
        drawing = ImageDraw.Draw(image)
        drawing.rectangle((0, 0, 960, 75), fill="#17324c")
        number = 15 + (index if change in {"number", "both"} else 0)
        text = f"Project progress\nProfit {number}%\nReport for the current quarter"
        drawing.multiline_text((45, 130), text, fill="#172a40", spacing=22)
        bar_end = 291 + (index * 8 if change in {"bar", "both"} else 0)
        drawing.rectangle((45, 340, bar_end, 410), fill="#347caa")
        stamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=index * 30)
        lines = [{"text": text, "confidence": .99, "polygon": []}] if with_ocr else []
        sources.append(SourceFrame(index + 1, "recording", stamp.isoformat(),
                                   stamp.isoformat(), image, lines, []))
    changed = np.any(np.asarray(sources[0].image) != np.asarray(sources[1].image), axis=2)
    assert np.mean(changed) < .003  # The previous whole-screen rule erased these changes.

    pages = render_document_pages(sources, True)

    assert len(pages) == 2
    assert [page.source_event_ids for page in pages] == [[1], [2]]
    for page, source in zip(pages, sources, strict=True):
        assert np.array_equal(np.asarray(page.image), np.asarray(source.image))
        if with_ocr:
            assert source.lines[0]["text"] in _all_text([page])


def test_duplicate_images_keep_ocr_without_polygon_geometry():
    first = _source(1, 0, 0, polygons=False)
    second = _source(2, 0, 10, polygons=False)
    pages = render_document_pages([first, second], True)

    assert len(pages) == 1
    assert pages[0].source_event_ids == [1, 2]
    assert _all_text(pages) == [line["text"] for line in first.lines]
