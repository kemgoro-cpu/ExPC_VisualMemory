from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ai import EmbeddingProvider
from .db import Database

TOKEN_RE = re.compile(r"[\w\-\.]+", re.UNICODE)


def safe_fts_query(query: str) -> str:
    tokens = [token.replace('"', '""') for token in TOKEN_RE.findall(query) if len(token) >= 3]
    return " OR ".join(f'"{token}"' for token in tokens[:32])


@dataclass(slots=True)
class SearchResult:
    event: dict[str, Any]
    score: float
    exact_rank: int | None = None
    semantic_rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.event,
            "score": self.score,
            "exact_rank": self.exact_rank,
            "semantic_rank": self.semantic_rank,
        }


class SearchEngine:
    def __init__(self, db: Database, embeddings: EmbeddingProvider):
        self.db = db
        self.embeddings = embeddings
        self._cache_lock = threading.Lock()
        # signature = (dimension, count, max_id)
        self._cache_signature: tuple[int, int, int] | None = None
        self._cache_rows: list[Any] = []
        self._cache_matrix: np.ndarray | None = None

    def search(
        self,
        query: str = "",
        start: str | None = None,
        end: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        if not query.strip():
            where, params = self._time_clause(start, end)
            rows = self.db.fetchall(
                f"SELECT * FROM screen_event {where} ORDER BY started_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            )
            return [
                self._row(row) | {"score": 0.0, "exact_rank": None, "semantic_rank": None} for row in rows
            ]

        exact = self._exact(query, start, end, 200)
        semantic = self._semantic(query, start, end, 200)
        scores: dict[int, SearchResult] = {}
        for rank, row in enumerate(exact, 1):
            event_id = int(row["id"])
            scores[event_id] = SearchResult(self._row(row), 1.0 / (60 + rank), exact_rank=rank)
        for rank, row in enumerate(semantic, 1):
            event_id = int(row["id"])
            contribution = 1.0 / (60 + rank)
            if event_id in scores:
                scores[event_id].score += contribution
                scores[event_id].semantic_rank = rank
            else:
                scores[event_id] = SearchResult(self._row(row), contribution, semantic_rank=rank)
        ordered = sorted(scores.values(), key=lambda item: (-item.score, item.event["started_at"]))
        return [item.to_dict() for item in ordered[offset : offset + limit]]

    def timeline_count(self, start: str | None = None, end: str | None = None) -> int:
        where, params = self._time_clause(start, end)
        row = self.db.fetchone(f"SELECT COUNT(*) AS count FROM screen_event {where}", params)
        return int(row["count"] if row else 0)

    def _exact(self, query: str, start: str | None, end: str | None, limit: int):
        expression = safe_fts_query(query)
        if not expression:
            return self._like_exact(query, start, end, limit)
        conditions = ["screen_event_fts MATCH ?"]
        params: list[Any] = [expression]
        if start:
            conditions.append("e.ended_at >= ?")
            params.append(start)
        if end:
            conditions.append("e.started_at <= ?")
            params.append(end)
        params.append(limit)
        fts_rows = self.db.fetchall(
            f"""
            SELECT e.*, bm25(screen_event_fts) AS bm25_score
            FROM screen_event_fts JOIN screen_event e ON e.id=screen_event_fts.rowid
            WHERE {" AND ".join(conditions)}
            ORDER BY bm25_score ASC LIMIT ?
            """,
            params,
        )
        # トライグラムFTSは3文字未満のトークンを拾えないため、
        # そのようなトークンが混在するクエリではLIKE検索の結果もマージする
        short_tokens = [token for token in TOKEN_RE.findall(query) if len(token) < 3]
        if not short_tokens:
            return fts_rows
        merged = list(fts_rows)
        seen_ids = {int(row["id"]) for row in fts_rows}
        for token in short_tokens:
            for row in self._like_exact(token, start, end, limit):
                row_id = int(row["id"])
                if row_id not in seen_ids:
                    seen_ids.add(row_id)
                    merged.append(row)
        return merged[:limit]

    def _like_exact(self, query: str, start: str | None, end: str | None, limit: int):
        value = query.strip()
        if not value:
            return []
        conditions = ["ocr_text LIKE ? ESCAPE '\\'"]
        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        params: list[Any] = [f"%{escaped}%"]
        if start:
            conditions.append("ended_at >= ?")
            params.append(start)
        if end:
            conditions.append("started_at <= ?")
            params.append(end)
        params.append(limit)
        return self.db.fetchall(
            f"""SELECT *, 0.0 AS bm25_score FROM screen_event
                WHERE {" AND ".join(conditions)} ORDER BY started_at DESC LIMIT ?""",
            params,
        )

    def _semantic(self, query: str, start: str | None, end: str | None, limit: int):
        vector = self.embeddings.encode_query(query)
        if vector is None:
            return []
        query_vector = np.asarray(vector, dtype=np.float32)
        rows, matrix = self._embedding_cache(query_vector.size)
        if matrix is None or not rows:
            return []
        scores = matrix @ query_vector
        eligible = np.ones(len(rows), dtype=bool)
        if start:
            eligible &= np.asarray([row["ended_at"] >= start for row in rows])
        if end:
            eligible &= np.asarray([row["started_at"] <= end for row in rows])
        indexes = np.flatnonzero(eligible)
        if indexes.size == 0:
            return []
        eligible_scores = scores[indexes]
        count = min(limit, indexes.size)
        if count < indexes.size:
            local = np.argpartition(eligible_scores, -count)[-count:]
            indexes = indexes[local]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        return [rows[int(index)] for index in indexes[:limit] if math.isfinite(float(scores[index]))]

    def _embedding_cache(self, dimension: int) -> tuple[list[Any], np.ndarray | None]:
        signature_row = self.db.fetchone(
            """SELECT COUNT(*) AS count, COALESCE(MAX(id), 0) AS max_id
               FROM screen_event WHERE embedding IS NOT NULL"""
        )
        count = int(signature_row["count"])
        max_id = int(signature_row["max_id"])
        with self._cache_lock:
            if self._cache_signature == (dimension, count, max_id):
                return self._cache_rows, self._cache_matrix

            cached_dimension, cached_count, cached_max_id = self._cache_signature or (None, 0, 0)
            can_append = cached_dimension == dimension and cached_max_id <= max_id
            if can_append:
                # 前回のmax_id以降に増えた行だけを追加ロードする(録画中の全再構築を避ける)
                new_rows = self.db.fetchall(
                    """SELECT * FROM screen_event
                       WHERE embedding IS NOT NULL AND embedding_dim=? AND id > ?
                       ORDER BY id""",
                    (dimension, cached_max_id),
                )
                appended_rows, appended_vectors = self._vectorize(new_rows, dimension)
                if cached_count + len(appended_rows) == count:
                    rows = self._cache_rows + appended_rows
                    if appended_vectors:
                        matrix = (
                            np.vstack([self._cache_matrix, *appended_vectors])
                            if self._cache_matrix is not None
                            else np.ascontiguousarray(np.vstack(appended_vectors), dtype=np.float32)
                        )
                    else:
                        matrix = self._cache_matrix
                    self._cache_rows = rows
                    self._cache_matrix = matrix
                    self._cache_signature = (dimension, count, max_id)
                    return self._cache_rows, self._cache_matrix
                # 途中の行が削除されるなどappendだけで説明できない差分があれば全再構築にフォールバック

            raw_rows = self.db.fetchall(
                "SELECT * FROM screen_event WHERE embedding IS NOT NULL AND embedding_dim=? ORDER BY id",
                (dimension,),
            )
            rows, vectors = self._vectorize(raw_rows, dimension)
            self._cache_rows = rows
            self._cache_matrix = (
                np.ascontiguousarray(np.vstack(vectors), dtype=np.float32) if vectors else None
            )
            self._cache_signature = (dimension, count, max_id)
            return self._cache_rows, self._cache_matrix

    @staticmethod
    def _vectorize(raw_rows: list[Any], dimension: int) -> tuple[list[Any], list[np.ndarray]]:
        rows: list[Any] = []
        vectors: list[np.ndarray] = []
        for row in raw_rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32)
            if vector.size != dimension:
                continue
            norm = float(np.linalg.norm(vector))
            if not norm:
                continue
            rows.append(row)
            vectors.append(vector / norm)
        return rows, vectors

    @staticmethod
    def _time_clause(start: str | None, end: str | None, prefix: str = "WHERE") -> tuple[str, list[str]]:
        conditions: list[str] = []
        params: list[str] = []
        if start:
            conditions.append("ended_at >= ?")
            params.append(start)
        if end:
            conditions.append("started_at <= ?")
            params.append(end)
        return (f"{prefix} {' AND '.join(conditions)}" if conditions else "", params)

    @staticmethod
    def _row(row) -> dict[str, Any]:
        data = dict(row)
        data.pop("embedding", None)
        text = data.get("ocr_text", "")
        data["ocr_excerpt"] = text[:500]
        return data

    def event_with_neighbors(self, event_id: int, count: int = 2) -> dict[str, Any] | None:
        event = self.db.fetchone("SELECT * FROM screen_event WHERE id=?", (event_id,))
        if not event:
            return None
        before = self.db.fetchall(
            "SELECT * FROM screen_event WHERE started_at < ? ORDER BY started_at DESC LIMIT ?",
            (event["started_at"], count),
        )
        after = self.db.fetchall(
            "SELECT * FROM screen_event WHERE started_at > ? ORDER BY started_at ASC LIMIT ?",
            (event["started_at"], count),
        )
        return {
            "event": self._row(event),
            "before": [self._row(row) for row in reversed(before)],
            "after": [self._row(row) for row in after],
        }
