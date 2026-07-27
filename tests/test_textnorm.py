from visual_memory.ai import OcrLine, OcrResult
from visual_memory.textnorm import (
    clean_ocr_lines,
    line_key,
    looks_like_chrome_noise,
    normalize_ocr_result,
    normalize_ocr_text,
)


def _line(text: str, center_y: float | None = None) -> dict:
    """テスト用の行データを作る。center_y は 0..1 の縦位置(Noneなら座標なし)。"""
    if center_y is None:
        return {"text": text, "polygon": []}
    y = center_y * 1000  # 高さ1000pxのフレームを前提にする
    return {"text": text, "polygon": [[0, y - 5], [100, y - 5], [100, y + 5], [0, y + 5]]}


def test_kouki_radical_normalized_to_kanji():
    assert normalize_ocr_text("⽇本語") == "日本語"
    assert normalize_ocr_text("テス⼘") == "テスト"  # 康熙部首"⼘"(卜の部首)


def test_katakana_context_kanji_confusion_fixed():
    assert normalize_ocr_text("テス卜") == "テスト"
    assert normalize_ocr_text("モジュ一ル") == "モジュール"
    assert normalize_ocr_text("タ-ミナル") == "ターミナル"


def test_non_katakana_context_kanji_is_not_touched():
    # 「一」がカタカナに挟まれていない場合は誤変換しない
    assert normalize_ocr_text("一番") == "一番"
    assert normalize_ocr_text("力を合わせる") == "力を合わせる"


def test_katakana_word_followed_by_hiragana_particle_is_fixed():
    # ひらがなの助詞は頻繁にカタカナ名詞へ続くため、誤変換の危険側とはみなさない
    assert normalize_ocr_text("テス卜を実行しました") == "テストを実行しました"


def test_fullwidth_alnum_and_parens_normalized_to_halfwidth():
    assert normalize_ocr_text("表示（Ｖ）") == "表示(V)"
    assert normalize_ocr_text("ＡＢＣ１２３") == "ABC123"


def test_newlines_are_preserved_and_do_not_bleed_across_lines():
    value = "タ\nーミナル"
    assert normalize_ocr_text(value) == "タ\nーミナル"


def test_empty_string_returns_empty():
    assert normalize_ocr_text("") == ""


def test_normalize_ocr_result_normalizes_text_and_each_line():
    result = OcrResult(
        text="テス卜\n⽇付",
        confidence=0.9,
        lines=[
            OcrLine(text="テス卜", confidence=0.9, polygon=[[0, 0]]),
            OcrLine(text="⽇付", confidence=0.8, polygon=[[1, 1]]),
        ],
    )
    normalized = normalize_ocr_result(result)
    assert normalized.text == "テスト\n日付"
    assert normalized.lines[0].text == "テスト"
    assert normalized.lines[0].polygon == [[0, 0]]  # polygonは不変
    assert normalized.lines[1].text == "日付"
    assert normalized.confidence == 0.9


# --- looks_like_chrome_noise: 文字パターンによるUI行の判定 ---


def test_looks_like_chrome_noise_detects_combined_menu_bar_line():
    assert looks_like_chrome_noise("表示(V) 移動(G) 実行(R) ターミナル(T) ヘルプ(H)")


def test_looks_like_chrome_noise_detects_single_known_menu_item():
    assert looks_like_chrome_noise("表示(V)")
    assert looks_like_chrome_noise("ヘルプ(H)")


def test_looks_like_chrome_noise_does_not_flag_ordinary_text_with_parens():
    assert not looks_like_chrome_noise("結果(暫定)を確認する")
    assert not looks_like_chrome_noise("四半期の売上高は前年比12.5%増")


# --- clean_ocr_lines: 索引前のノイズ除去 ---


def test_clean_ocr_lines_removes_clock_date_and_slide_counter():
    lines = [_line("14:32"), _line("2026/07/27"), _line("スライド 3/20"), _line("会議の議事録")]
    assert clean_ocr_lines(lines, height=1000) == ["会議の議事録"]


def test_clean_ocr_lines_deduplicates_within_one_frame():
    lines = [_line("設計方針の見直し"), _line("設計方針の見直し"), _line("次の課題")]
    assert clean_ocr_lines(lines, height=1000) == ["設計方針の見直し", "次の課題"]


def test_clean_ocr_lines_drops_repeated_lines_only_at_screen_edges():
    """常時表示の行でも、画面中央にあるなら本文として残す(位置ルールの検証)。"""
    repeated = {line_key("エクスプローラー")}
    top = [_line("エクスプローラー", center_y=0.05)]
    bottom = [_line("エクスプローラー", center_y=0.95)]
    middle = [_line("エクスプローラー", center_y=0.50)]

    assert clean_ocr_lines(top, height=1000, repeated_keys=repeated) == []
    assert clean_ocr_lines(bottom, height=1000, repeated_keys=repeated) == []
    assert clean_ocr_lines(middle, height=1000, repeated_keys=repeated) == ["エクスプローラー"]


def test_clean_ocr_lines_keeps_edge_lines_that_are_not_repeated():
    """繰り返し出現していない行は、画面下端にあっても本文として残す。"""
    lines = [_line("結論: 案Bを採用する", center_y=0.95)]
    assert clean_ocr_lines(lines, height=1000, repeated_keys=set()) == ["結論: 案Bを採用する"]


def test_clean_ocr_lines_skips_position_rule_without_coordinates_or_height():
    repeated = {line_key("エクスプローラー")}
    # 座標がない行は位置で判定できない
    assert clean_ocr_lines([_line("エクスプローラー")], 1000, repeated) == ["エクスプローラー"]
    # 高さが不明(0)なら比率を計算できない
    assert clean_ocr_lines([_line("エクスプローラー", 0.95)], 0, repeated) == ["エクスプローラー"]
