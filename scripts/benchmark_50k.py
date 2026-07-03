from __future__ import annotations

import argparse
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from visual_memory.config import Settings
from visual_memory.db import Database
from visual_memory.search import SearchEngine


class FixedEmbedding:
    name = "benchmark"
    available = True
    reason = None

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.query = np.ones(dimension, dtype=np.float32)
        self.query /= np.linalg.norm(self.query)

    def encode_query(self, text: str):
        return self.query

    def encode_document(self, text: str):
        return self.query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=int, default=50_000)
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--target-seconds", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="visual-memory-benchmark-"))
    settings = Settings(data_dir=root)
    db = Database(settings.database_path)
    db.initialize()
    started = datetime(2026, 1, 1, tzinfo=UTC)
    db.execute(
        """INSERT INTO capture_session(id,source_name,started_at,status)
           VALUES('bench','generated',?,'stopped')""",
        (started.isoformat(),),
    )
    rng = np.random.default_rng(42)
    rows = []
    for index in range(args.events):
        vector = rng.normal(size=args.dimension).astype(np.float32)
        vector /= np.linalg.norm(vector)
        stamp = (started + timedelta(seconds=index)).isoformat()
        rows.append(
            (
                "bench",
                stamp,
                stamp,
                "frame.webp",
                "thumb.webp",
                f"{index:016x}",
                0.1,
                1920,
                1080,
                "stable",
                f"event {index}",
                0.99,
                vector.tobytes(),
                args.dimension,
                stamp,
                stamp,
            )
        )
    db.executemany(
        """INSERT INTO screen_event(
            session_id,started_at,ended_at,frame_path,thumbnail_path,phash,change_score,width,height,
            event_kind,ocr_text,ocr_confidence,embedding,embedding_dim,created_at,processed_at
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    engine = SearchEngine(db, FixedEmbedding(args.dimension))
    before = time.perf_counter()
    result = engine.search("remembered concept", limit=5)
    cold = time.perf_counter() - before
    before = time.perf_counter()
    engine.search("another concept", limit=5)
    warm = time.perf_counter() - before
    print(f"events={args.events} cold={cold:.3f}s warm={warm:.3f}s top={result[0]['id']}")
    raise SystemExit(0 if warm <= args.target_seconds else 1)


if __name__ == "__main__":
    main()
