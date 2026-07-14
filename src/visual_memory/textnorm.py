from __future__ import annotations

import re
import unicodedata

from .ai import OcrLine, OcrResult

# OCRは見た目が似た文字を取り違えることがある(康熙部首の"⽇"と漢字の"日"、
# カタカナの中に紛れ込む同形の漢字など)。「隣接する文字が漢字でなければ
# (=単語の切れ目・ひらがなの助詞・カタカナそのものであれば)変換してよい」という
# ルールで補正し、「一番」「力士」のような漢字の複合語を巻き込まないようにする。
# 少なくとも片側はカタカナである必要がある(句読点だけに挟まれた文字は変換しない)。
# ひらがなは助詞として頻繁に隣接する(例:「テス卜を」)ため危険側に含めない。
_KANA_CONFUSABLES: dict[str, str] = {
    "卜": "ト",
    "一": "ー",
    "二": "ニ",
    "力": "カ",
    "工": "エ",
    "夕": "タ",
    "口": "ロ",
    "八": "ハ",
    "-": "ー",  # ASCIIハイフンとカタカナ長音の混同(例: "タ-ミナル"→"ターミナル")
}
_KANA_RE = re.compile(r"[ァ-ヶー]")  # ァ-ヶ, ー
_BLOCKING_RE = re.compile(r"[一-龠]")  # 漢字のみ(ひらがなは助詞として安全なため除外)


def _fix_kana_confusables(text: str) -> str:
    chars = list(text)
    for index, char in enumerate(chars):
        if char not in _KANA_CONFUSABLES:
            continue
        before = chars[index - 1] if index > 0 else ""
        after = chars[index + 1] if index + 1 < len(chars) else ""
        before_ok = not before or _KANA_RE.match(before) or not _BLOCKING_RE.match(before)
        after_ok = not after or _KANA_RE.match(after) or not _BLOCKING_RE.match(after)
        has_kana_neighbor = bool(_KANA_RE.match(before) or _KANA_RE.match(after))
        if before_ok and after_ok and has_kana_neighbor:
            chars[index] = _KANA_CONFUSABLES[char]
    return "".join(chars)


def normalize_ocr_text(value: str) -> str:
    """OCR結果の文字ゆらぎを補正する(改行は保持する)。

    既知の限界: カタカナ語の直後に実在の漢字が続く稀なケース(例:「テスト力」の「力」)
    は誤って変換されうる。小書き文字(ュ/ユ等)の混同は文脈だけでは判別できないため
    補正しない。頻出パターン(単語末尾の誤字)を優先した設計上のトレードオフ。
    """
    if not value:
        return value
    normalized = unicodedata.normalize("NFKC", value)
    return _fix_kana_confusables(normalized)


def normalize_ocr_result(result: OcrResult) -> OcrResult:
    """OcrResult全体(本文+行ごとのテキスト)を正規化した新しいインスタンスを返す。"""
    return OcrResult(
        text=normalize_ocr_text(result.text),
        confidence=result.confidence,
        lines=[
            OcrLine(text=normalize_ocr_text(line.text), confidence=line.confidence, polygon=line.polygon)
            for line in result.lines
        ],
    )
