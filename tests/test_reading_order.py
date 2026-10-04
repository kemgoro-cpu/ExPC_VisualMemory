from visual_memory.reading_order import polygon_box, sort_in_reading_order


def _line(text: str, left: float, top: float, right: float, bottom: float) -> dict:
    """テスト用に、四隅の座標を持つ1行分のOCRデータを作る。"""
    return {
        "text": text,
        "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
    }


def _order(lines: list[dict]) -> list[str]:
    ordered = sort_in_reading_order(lines, lambda line: polygon_box(line.get("polygon")))
    return [line["text"] for line in ordered]


def test_two_columns_are_read_one_column_at_a_time():
    """2段組みは、左の段を読み切ってから右の段に移る。"""
    # OCRは縦位置が近い順に返すため、左右が1行ずつ交互に並んだ状態を入力にする
    lines = [
        _line("左1", 0, 0, 100, 20),
        _line("右1", 150, 0, 250, 20),
        _line("左2", 0, 30, 100, 50),
        _line("右2", 150, 30, 250, 50),
        _line("左3", 0, 60, 100, 80),
    ]
    assert _order(lines) == ["左1", "左2", "左3", "右1", "右2"]


def test_full_width_heading_comes_before_the_columns_below_it():
    """段組みを跨ぐ見出しは、下の段組みより先に出る。"""
    # 見出しの下は行間(10px)より広く空ける。実際のレイアウトでも見出しの下には
    # 本文の行間より大きな余白が入り、そこが「意味のある区切り」になる
    lines = [
        _line("章の見出し", 0, 0, 250, 20),
        _line("右1", 150, 45, 250, 65),
        _line("左1", 0, 45, 100, 65),
        _line("左2", 0, 75, 100, 95),
        _line("右2", 150, 75, 250, 95),
    ]
    assert _order(lines) == ["章の見出し", "左1", "左2", "右1", "右2"]


def test_single_column_keeps_top_to_bottom_order():
    lines = [
        _line("1行目", 0, 0, 250, 20),
        _line("2行目", 0, 30, 250, 50),
        _line("3行目", 0, 60, 250, 80),
    ]
    assert _order(lines) == ["1行目", "2行目", "3行目"]


def test_words_on_the_same_line_are_not_treated_as_separate_columns():
    """同じ行の中の単語間の空白を、段の切れ目と取り違えない。"""
    lines = [
        _line("見出し", 0, 0, 40, 20),
        _line("補足", 55, 0, 100, 20),  # 15pxの空き = 行の高さより狭い
        _line("本文の行", 0, 30, 100, 50),
    ]
    assert _order(lines) == ["見出し", "補足", "本文の行"]


def test_lines_without_coordinates_keep_their_original_order():
    """座標が取れない行が混ざるときは、判断材料が無いので並べ替えない。"""
    lines = [
        _line("右1", 150, 0, 250, 20),
        {"text": "座標なし", "polygon": []},
        _line("左1", 0, 0, 100, 20),
    ]
    assert _order(lines) == ["右1", "座標なし", "左1"]


def test_three_columns_are_read_left_to_right():
    # 段の間は40px(行の高さ20pxの2倍)。実際の段組みもこの程度は空いている
    lines = [
        _line("中1", 140, 0, 240, 20),
        _line("右1", 280, 0, 380, 20),
        _line("左1", 0, 0, 100, 20),
        _line("左2", 0, 30, 100, 50),
        _line("中2", 140, 30, 240, 50),
    ]
    assert _order(lines) == ["左1", "左2", "中1", "中2", "右1"]
