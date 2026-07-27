"""保存済みイベントの screen_event.ocr_text_clean を作り直すスクリプト。

画面をフル解像度で取り込んでいるため、本文と一緒にタスクバー・時計・メニューバーなどの
「常時表示のUI」もOCRされている。これらは全フレームに共通して現れるので、残したまま
全文検索の索引や埋め込みを作ると検索精度が落ちる。

索引処理(processing.EventProcessor)は1フレームずつしか見られないので直近フレームの
統計で近似しているが、このスクリプトはセッション全体を見渡せるため、より正確に
「常時表示の行」を判定して洗浄テキストを作り直せる。

OCRの再実行はしない(evidence に保存済みの行データを使う)。埋め込みも再計算しない。
実行前にアプリを終了し、visual-memory.sqlite3 をバックアップしてください。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from visual_memory.config import load_settings
from visual_memory.db import Database
from visual_memory.textnorm import clean_ocr_lines, line_key

# packs._clean_ocr_items と同じ「6割以上のフレームに出れば常時表示」という基準
REPEAT_RATIO = 0.6


def _extract_lines(metadata_json: str | None, ocr_text: str) -> list[dict[str, Any]]:
    """evidence の metadata_json から行データ(テキスト+座標)を取り出す。

    metadata_json が無い古いイベントでは座標が得られないため、生テキストを行分割した
    ものを返す。この場合は位置による判定が効かず、パターン判定のみが適用される。
    """
    try:
        lines = json.loads(metadata_json or "{}").get("lines") or []
    except (json.JSONDecodeError, AttributeError, TypeError):
        lines = []
    if lines:
        return [
            {"text": str(line.get("text", "")), "polygon": line.get("polygon") or []}
            for line in lines
        ]
    return [{"text": value, "polygon": []} for value in (ocr_text or "").splitlines()]


def _load_session_events(db: Database, session_id: str) -> list[tuple[int, int, str, list[dict]]]:
    """セッション内の全イベントを (id, height, 現在の洗浄テキスト, 行データ) で返す。"""
    # 1イベントに複数の 'ocr' evidence が残ることがある(再処理したとき)ので、
    # 最も新しいもの(idが最大)だけを使う
    rows = db.fetchall(
        """
        SELECT e.id, e.height, e.ocr_text, e.ocr_text_clean, v.metadata_json
        FROM screen_event e
        LEFT JOIN evidence v ON v.id = (
            SELECT id FROM evidence WHERE event_id = e.id AND kind = 'ocr'
            ORDER BY id DESC LIMIT 1
        )
        WHERE e.session_id = ?
        ORDER BY e.started_at
        """,
        (session_id,),
    )
    return [
        (
            int(row["id"]),
            int(row["height"] or 0),
            str(row["ocr_text_clean"] or ""),
            # JSON文字列は保持せず、必要な項目だけに絞ってメモリを抑える
            _extract_lines(row["metadata_json"], str(row["ocr_text"] or "")),
        )
        for row in rows
    ]


def _repeated_keys(events: list[tuple[int, int, str, list[dict]]]) -> set[str]:
    """セッション内の6割以上のフレームに出現する行のキー集合を返す。"""
    frequency: Counter[str] = Counter()
    for _event_id, _height, _current, lines in events:
        frequency.update({line_key(str(line["text"])) for line in lines})
    threshold = max(2, math.ceil(len(events) * REPEAT_RATIO))
    return {key for key, count in frequency.items() if count >= threshold}


def backfill(db: Database, *, dry_run: bool) -> tuple[int, int]:
    """全セッションの ocr_text_clean を作り直す。 (変更件数, 全件数) を返す。"""
    session_ids = [
        str(row["session_id"])
        for row in db.fetchall("SELECT DISTINCT session_id FROM screen_event ORDER BY session_id")
    ]
    changed = 0
    total = 0
    for session_id in session_ids:
        events = _load_session_events(db, session_id)
        if not events:
            continue
        total += len(events)
        repeated = _repeated_keys(events)
        updates: list[tuple[str, int]] = []
        for event_id, height, current, lines in events:
            cleaned = "\n".join(clean_ocr_lines(lines, height, repeated))
            if cleaned != current:
                updates.append((cleaned, event_id))
        changed += len(updates)
        if updates and not dry_run:
            # トリガー(screen_event_au)がFTSインデックスを自動追従するため、
            # 別途FTSの再構築は不要
            db.executemany(
                "UPDATE screen_event SET ocr_text_clean=? WHERE id=?", updates
            )
    return changed, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="対象データディレクトリ(既定: 通常の保存先)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="変更件数だけを表示し、実際には更新しない"
    )
    parser.add_argument(
        "--reembed",
        action="store_true",
        help="洗浄後テキストで埋め込みを作り直す(未実装)",
    )
    args = parser.parse_args()

    if args.reembed:
        raise SystemExit(
            "--reembed はまだ実装されていません。"
            "個別のイベントはUIの「OCR・検索を再処理」から作り直せます。"
        )

    settings = load_settings(str(args.data_dir) if args.data_dir else None)
    if not settings.database_path.exists():
        raise SystemExit(f"Database not found: {settings.database_path}")

    print(f"database: {settings.database_path}")
    if not args.dry_run:
        print("アプリが起動中の場合は先に終了し、DBのバックアップを取ってください。")

    db = Database(settings.database_path)
    db.initialize()

    changed, total = backfill(db, dry_run=args.dry_run)

    prefix = "[dry-run] " if args.dry_run else ""
    suffix = "件が変更対象です" if args.dry_run else "件を更新しました"
    print(f"{prefix}screen_event: {total}件中 {changed}{suffix}")
    print(
        "embeddingは再計算していません。"
        "既存イベントのベクトルは洗浄前のテキストから作られたままです。"
    )


if __name__ == "__main__":
    main()
