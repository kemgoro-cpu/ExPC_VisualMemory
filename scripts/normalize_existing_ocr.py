from __future__ import annotations

import argparse
import json
from pathlib import Path

from visual_memory.config import load_settings
from visual_memory.db import Database
from visual_memory.textnorm import normalize_ocr_text


def _normalize_screen_events(db: Database, *, dry_run: bool) -> int:
    rows = db.fetchall("SELECT id, ocr_text FROM screen_event WHERE ocr_text != ''")
    changed = 0
    for row in rows:
        normalized = normalize_ocr_text(row["ocr_text"])
        if normalized == row["ocr_text"]:
            continue
        changed += 1
        if not dry_run:
            # トリガー(screen_event_au)がFTSインデックスを自動追従するため、
            # 別途FTSの再構築は不要
            db.execute("UPDATE screen_event SET ocr_text=? WHERE id=?", (normalized, row["id"]))
    return changed


def _normalize_evidence(db: Database, *, dry_run: bool) -> int:
    rows = db.fetchall("SELECT id, text, metadata_json FROM evidence WHERE kind='ocr'")
    changed = 0
    for row in rows:
        text = row["text"] or ""
        normalized_text = normalize_ocr_text(text)
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        lines = metadata.get("lines") or []
        normalized_lines = []
        lines_changed = False
        for line in lines:
            original = str(line.get("text", ""))
            normalized_line_text = normalize_ocr_text(original)
            if normalized_line_text != original:
                lines_changed = True
            normalized_lines.append({**line, "text": normalized_line_text})
        if normalized_text == text and not lines_changed:
            continue
        changed += 1
        if not dry_run:
            metadata["lines"] = normalized_lines
            db.execute(
                "UPDATE evidence SET text=?, metadata_json=? WHERE id=?",
                (normalized_text, json.dumps(metadata, ensure_ascii=False), row["id"]),
            )
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "保存済みのOCRテキストを正規化する(康熙部首の補正、カタカナ文脈での"
            "混同漢字補正、長音とハイフンの混同補正など)。"
            "実行前にアプリを終了し、visual-memory.sqlite3をバックアップしてください。"
        )
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="対象データディレクトリ(既定: 通常の保存先)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="変更件数だけを表示し、実際には更新しない"
    )
    args = parser.parse_args()

    settings = load_settings(str(args.data_dir) if args.data_dir else None)
    if not settings.database_path.exists():
        raise SystemExit(f"Database not found: {settings.database_path}")

    print(f"database: {settings.database_path}")
    if not args.dry_run:
        print("アプリが起動中の場合は先に終了し、DBのバックアップを取ってください。")

    db = Database(settings.database_path)
    db.initialize()

    event_changes = _normalize_screen_events(db, dry_run=args.dry_run)
    evidence_changes = _normalize_evidence(db, dry_run=args.dry_run)

    prefix = "[dry-run] " if args.dry_run else ""
    suffix = "件が変更対象です" if args.dry_run else "件を更新しました"
    print(f"{prefix}screen_event: {event_changes}{suffix}")
    print(f"{prefix}evidence: {evidence_changes}{suffix}")
    print("embeddingは再計算していません(表記ゆらぎ程度では意味ベクトルへの影響は軽微なため)。")


if __name__ == "__main__":
    main()
