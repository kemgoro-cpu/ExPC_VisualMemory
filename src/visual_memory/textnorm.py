from __future__ import annotations

import re
import unicodedata
from collections.abc import Container, Iterable, Mapping
from typing import Any

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


# --- ここから下: 画面のふち(タスクバー・メニューバー等)のノイズ行を落とすための判定 ---
#
# 画面をフル解像度でキャプチャしているため、本文と一緒にタスクバー・時計・リボン・
# タブバーといった「常時表示のUI」までOCRされてしまう。これらは全フレームに写るので、
# 埋め込みベクトルには共通成分として乗り、全文検索には毎回ヒットする。どちらも検索精度を
# 落とすため、索引を作る前に行単位で除去する。
# (画像そのものは証跡として残すため、切り抜きではなくテキスト側で落とす方針)

NOISE_LABELS = {
    "検索", "q 検索", "ファイル", "ホーム", "挿入", "描画", "デザイン",
    "画面切り替え", "アニメーション", "スライドショー", "record", "校閲",
    "表示", "ヘルプ", "acrobat", "共有", "日本語", "ノート", "表示設定",
    "コメント", "アクセシビリティ：検討が必要です",
    "アクセシビリティ: 検討が必要です",
}
TITLE_EXTENSIONS = (
    r"pptx?|docx?|xlsx?|pdf|py|ipynb|js|ts|tsx|jsx|java|cs|cpp|c|h"
    r"|md|txt|ya?ml|json|html?|css"
)
FILE_TITLE_PATTERN = re.compile(rf"(?i)(.+?\.(?:{TITLE_EXTENSIONS}))")

# 画面のふち: この帯に入る行だけを「常時表示なら除去」の対象にする。
# 中央付近に出る繰り返し表現(見出しなど)は本文なので残す。
OUTER_CHROME_TOP_RATIO = 0.14
OUTER_CHROME_BOTTOM_RATIO = 0.82

_ACCESS_KEY_ITEM = re.compile(r"([^\s()]+)\(([A-Za-z])\)")
_UI_TERMS = (
    "ファイル", "ホーム", "挿入", "描画", "デザイン", "画面切り替え",
    "アニメーション", "スライドショー", "record", "校閲", "表示", "ヘルプ",
    "acrobat", "共有", "ノート", "表示設定",
)


def normalize_ocr_line(value: str) -> str:
    """1行分のOCRテキストを正規化し、連続する空白を1つにまとめる。"""
    return re.sub(r"\s+", " ", normalize_ocr_text(value)).strip()


def looks_like_chrome_noise(value: str) -> bool:
    """行が「本文ではなくUIの部品」に見えるかを判定する。"""
    normalized = normalize_ocr_line(value)
    lowered = normalized.casefold()
    if not normalized or lowered in NOISE_LABELS:
        return True
    if len(normalized) <= 1:
        return True
    if re.fullmatch(r"[+×□○●△▽◇◆★☆※*%]+", normalized):
        return True
    # メニューバーの「表示(V) 移動(G) 実行(R)…」のようなアクセスキー付き項目が
    # 1行に複数並ぶ行はメニューバー全体とみなして除去する
    if len(_ACCESS_KEY_ITEM.findall(normalized)) >= 2:
        return True
    # 単独行でも、ラベル部分が既知のノイズ語彙(表示・ヘルプ等)に一致すれば除去する
    single_access_key = _ACCESS_KEY_ITEM.fullmatch(normalized)
    if single_access_key and single_access_key.group(1).casefold() in NOISE_LABELS:
        return True
    if sum(term in lowered for term in _UI_TERMS) >= 2:
        return True
    if FILE_TITLE_PATTERN.search(normalized) and any(
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


def line_key(value: str) -> str:
    """行を「同じ行かどうか」で突き合わせるためのキーに変換する。"""
    return normalize_ocr_line(value).casefold()


def clean_ocr_lines(
    lines: Iterable[Mapping[str, Any]],
    height: int,
    repeated_keys: Container[str] | None = None,
) -> list[str]:
    """OCRの行データからノイズを除いた行テキストの一覧を返す。

    Args:
        lines: ``{"text": str, "polygon": [[x, y], ...]}`` 形式の行データ。
            polygon は無くてもよい(その場合は位置による判定を行わない)。
        height: フレームの高さ(px)。0以下なら位置による判定を行わない。
        repeated_keys: 「複数フレームで繰り返し出現する行」の :func:`line_key` 集合。
            呼び出し側が数え方を決める(パック生成はページ群、索引処理は直近フレーム)。

    除去するもの:
        1. :func:`looks_like_chrome_noise` に一致する行(時計・日付・メニューバー等)
        2. 同一フレーム内で重複する行
        3. ``repeated_keys`` に含まれ、かつ画面の上端/下端の帯にある行
           (= タスクバーやタイトルバーのように常に同じ位置に出る行)
    """
    repeated = repeated_keys if repeated_keys is not None else frozenset()
    cleaned: list[str] = []
    seen: set[str] = set()
    for line in lines:
        text = normalize_ocr_line(str(line.get("text", "")))
        key = text.casefold()
        if key in seen or looks_like_chrome_noise(text):
            continue
        polygon = line.get("polygon") or []
        ys = [float(point[1]) for point in polygon if len(point) >= 2]
        if ys and height > 0 and key in repeated:
            center_y = (min(ys) + max(ys)) / 2 / height
            if center_y <= OUTER_CHROME_TOP_RATIO or center_y >= OUTER_CHROME_BOTTOM_RATIO:
                continue
        seen.add(key)
        cleaned.append(text)
    return cleaned
