from visual_memory.ai import OcrLine, OcrResult
from visual_memory.textnorm import normalize_ocr_result, normalize_ocr_text


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
