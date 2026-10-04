"""OCRが返した行を、人が読む順(段組みを考慮した順序)に並べ替える。

OCRエンジンは行をおおむね上から下へ返すが、それは「縦位置が近い行は同じ行」という
前提に立っている。画面が2段組みの表やカードになっていると、左の段と右の段の行が
1行ずつ交互に並び、文章が読めなくなる:

    「必要出力(走行抵抗+加速)」と「各ギヤで得られる利用   ← 左の段の1行目
    使用するデータはエンジン回転速度と全負荷(WOT)出力     ← 右の段の1行目
    可能出力(全開出力曲線)」のバランスにより、秒単位でギ   ← 左の段の2行目

そこで「XYカット」という古典的なレイアウト解析の手法で並べ直す。やることは単純で、
行を囲む四角形を眺めて「どの行も跨いでいない空白の帯」を探し、そこで切って再帰する:

    1. 縦の帯(段と段の間の余白)があれば、そこで左右に切る → 左を全部出してから右
    2. 無ければ横の帯(段落や見出しの間の余白)で上下に切る → 上から順に出す
    3. どちらも無ければ、上から下・左から右の順に並べる

縦を先に試すのがポイント。ページ全体に渡る見出しがあると、その行が縦の帯を塞ぐので
縦には切れず、自動的に横の切り方(見出しと本文を分ける)が選ばれる。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# 段と段の間の余白と認めるための最小の幅。
# 行の高さの何倍か、ブロック全体の幅の何割か、の大きいほうを使う。
# 単語の間の空白を「段の切れ目」と誤認しないよう、1行の高さより広いことを求める。
_COLUMN_GAP_HEIGHT_RATIO = 1.2
_COLUMN_GAP_WIDTH_RATIO = 0.015

# 上下に切るときの最小の余白。ふつうの行間(行の高さの2〜5割)では切らず、
# 段落や見出しの区切りといった「意味のある余白」でだけ切るために大きめに取る。
# ここを小さくすると、段組みを見つける前に1行ずつのバンドへ切り刻んでしまい、
# 左右の段が交互に並ぶ元の問題に逆戻りする。
_ROW_GAP_HEIGHT_RATIO = 0.8

# 再帰の深さの上限。想定外のレイアウトで延々と切り続けないための保険
_MAX_DEPTH = 12

# (left, top, right, bottom)
Box = tuple[float, float, float, float]


def polygon_box(polygon: Any) -> Box | None:
    """多角形の座標から、それを囲む四角形を求める。座標が無ければ None。"""
    xs = [float(point[0]) for point in polygon or [] if len(point) >= 2]
    ys = [float(point[1]) for point in polygon or [] if len(point) >= 2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _group_by_gaps[T](
    entries: list[tuple[Box, T]], axis: int, minimum_gap: float
) -> list[list[tuple[Box, T]]]:
    """1つの軸に投影して、``minimum_gap`` 以上の空白があるところで塊に分ける。

    Args:
        entries: (四角形, 元の要素) の一覧。
        axis: 0なら横方向(左右に分ける)、1なら縦方向(上下に分ける)。
        minimum_gap: 空白と認める最小の幅(px)。

    Returns:
        分けた塊の一覧(座標の小さいほうから)。分けられなければ1個だけの一覧。
    """
    start_index, end_index = (0, 2) if axis == 0 else (1, 3)
    ordered = sorted(entries, key=lambda entry: entry[0][start_index])
    groups: list[list[tuple[Box, T]]] = []
    current: list[tuple[Box, T]] = []
    # これまでに見た四角形の「終端の最大値」。ここから次の始端までが空白になる
    reach = float("-inf")
    for entry in ordered:
        start, end = entry[0][start_index], entry[0][end_index]
        if current and start - reach >= minimum_gap:
            groups.append(current)
            current = []
            reach = float("-inf")
        current.append(entry)
        reach = max(reach, end)
    if current:
        groups.append(current)
    return groups


def _sorted_by_position[T](entries: list[tuple[Box, T]]) -> list[tuple[Box, T]]:
    """上から下、同じ高さなら左から右に並べる(これ以上切れないときの並べ方)。"""
    return sorted(entries, key=lambda entry: (entry[0][1], entry[0][0]))


def _cut[T](entries: list[tuple[Box, T]], line_height: float, depth: int) -> list[tuple[Box, T]]:
    if len(entries) <= 1 or depth >= _MAX_DEPTH:
        return _sorted_by_position(entries)

    width = max(box[2] for box, _item in entries) - min(box[0] for box, _item in entries)
    column_gap = max(line_height * _COLUMN_GAP_HEIGHT_RATIO, width * _COLUMN_GAP_WIDTH_RATIO)

    # 1. 段組み(縦の帯)で切れるならそちらを優先する
    columns = _group_by_gaps(entries, axis=0, minimum_gap=column_gap)
    if len(columns) > 1:
        return [
            entry for column in columns for entry in _cut(column, line_height, depth + 1)
        ]

    # 2. 切れなければ横の帯で上下に分ける
    rows = _group_by_gaps(entries, axis=1, minimum_gap=line_height * _ROW_GAP_HEIGHT_RATIO)
    if len(rows) > 1:
        return [entry for row in rows for entry in _cut(row, line_height, depth + 1)]

    return _sorted_by_position(entries)


def sort_in_reading_order[T](items: Sequence[T], box_of: Callable[[T], Box | None]) -> list[T]:
    """``items`` を読む順に並べ替えて返す。

    Args:
        items: 並べ替えたい要素の一覧。
        box_of: 要素から (left, top, right, bottom) を取り出す関数。
            座標が取れない要素があるときは並べ替えを行わない(下記)。

    Returns:
        並べ替えた新しい一覧。座標を持たない要素が1つでもあれば、判断材料が
        揃わないため元の順序をそのまま返す(座標なしの行を末尾へ追いやったり、
        位置0とみなして先頭に並べたりするより、元の順序のほうが安全なため)。
    """
    entries: list[tuple[Box, T]] = []
    for item in items:
        box = box_of(item)
        if box is None:
            return list(items)
        entries.append((box, item))
    if len(entries) <= 1:
        return list(items)

    line_height = _median([box[3] - box[1] for box, _item in entries])
    if line_height <= 0:
        return list(items)
    return [item for _box, item in _cut(entries, line_height, 0)]
