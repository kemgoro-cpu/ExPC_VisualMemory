from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from .ai import EmbeddingProvider, OcrProvider, OcrResult, order_ocr_result
from .capture import FrameCandidate, iso
from .db import Database
from .imaging import Region, changed_fraction, hash_distance, perceptual_hash, region_changed_fraction, to_bgr
from .storage import Storage
from .textnorm import clean_ocr_lines, line_key, normalize_ocr_result

LOGGER = logging.getLogger(__name__)


class ChromeLineTracker:
    """記録中のセッションで「繰り返し出現する行」を追跡する。

    タスクバーのアプリ名のように、文字のパターンだけでは本文と区別できないが
    「毎フレーム同じ位置に出続ける」ノイズを見つけるための仕組み。コンテキストパック
    生成側(packs._clean_ocr_items)はページ群全体を一度に見て頻度を数えられるが、
    索引処理は1フレームずつ流れてくるため、直近フレームの統計で近似する。

    記録は同時に1セッションしか動かないため、セッションが変わったら統計は捨てる。
    """

    def __init__(self, window: int = 300, warmup: int = 20, ratio: float = 0.6):
        self.window = window  # この件数を超えたらカウントを減衰させる(メモリ抑制)
        self.warmup = warmup  # これ未満の観測数では判定しない(初期の過剰除去を防ぐ)
        self.ratio = ratio  # 観測数に対する出現率がこれ以上なら「常時表示」とみなす
        self._session_id: str | None = None
        self._counts: Counter[str] = Counter()
        self._observed = 0
        self._lock = threading.Lock()

    def observe(self, session_id: str, keys: set[str]) -> set[str]:
        """1フレーム分の行キーを記録し、「常時表示」と判定した行キーの集合を返す。"""
        with self._lock:
            if session_id != self._session_id:
                self._session_id = session_id
                self._counts = Counter()
                self._observed = 0
            if self._observed >= self.window:
                self._decay()
            self._counts.update(keys)
            self._observed += 1
            if self._observed < self.warmup:
                return set()
            threshold = self._observed * self.ratio
            return {key for key, count in self._counts.items() if count >= threshold}

    def _decay(self) -> None:
        """カウントを半減させ、古い情報を薄めつつ出現率の連続性を保つ。

        丸ごとリセットすると判定が毎回ウォームアップからやり直しになるため、
        半減させて「めったに出ない行」だけを落とす。
        """
        for key, count in list(self._counts.items()):
            halved = count // 2
            if halved:
                self._counts[key] = halved
            else:
                del self._counts[key]
        self._observed //= 2


class LatestQueue[T]:
    """Bounded queue that retains the newest visual state when processing falls behind."""

    def __init__(self, maxsize: int = 8, on_drop: Callable[[T, str], None] | None = None):
        self.maxsize = maxsize
        self._items: deque[T] = deque()
        self._condition = threading.Condition()
        self.closed = False
        self.drops = 0
        self.on_drop = on_drop

    def put(self, item: T) -> bool:
        dropped = False
        with self._condition:
            if self.closed:
                if self.on_drop:
                    self.on_drop(item, "Storage processor is stopped")
                return False
            if len(self._items) >= self.maxsize:
                removed = self._items.popleft()
                if self.on_drop:
                    self.on_drop(removed, "Capture was dropped because the storage queue is full")
                self.drops += 1
                dropped = True
            self._items.append(item)
            self._condition.notify()
        return not dropped

    def get(self, timeout: float | None = None) -> T | None:
        end = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items and not self.closed:
                remaining = None if end is None else max(0.0, end - time.monotonic())
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            return self._items.popleft() if self._items else None

    def close(self) -> None:
        with self._condition:
            self.closed = True
            self._condition.notify_all()

    def reopen(self) -> None:
        with self._condition:
            self.closed = False

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)


@dataclass(slots=True)
class ProcessingItem:
    session_id: str
    candidate: FrameCandidate


@dataclass(slots=True)
class ProcessorStatus:
    state: str = "stopped"
    queue_depth: int = 0
    queue_drops: int = 0
    processed: int = 0
    failures: int = 0
    last_error: str | None = None
    last_processed_at: str | None = None
    skipped_duplicates: int = 0
    saved_count: int = 0
    session_saved_count: int = 0
    last_saved_at: str | None = None
    last_saved_kind: str | None = None
    last_saved_event_id: int | None = None
    last_saved_session_id: str | None = None


# --- 同じ画面の保存を避けるための判定 ---
#
# マウスカーソルの移動やボタンのホバー効果でも変化検出は反応するため、内容としては
# 同じ画面が何枚も保存される(実データで連続イベントの15%が該当した)。ここで落とす。
#
# 判定は「画素の変化がごくわずか」と「pHashがほぼ一致」の両方が成り立つときだけ。
# pHashは32x32のDCTを8x8に畳んだ粗い指紋なので、小さな文字の変更を見落とす。
# 画素レベルの判定(changed_fraction)と併用しないと、本当に内容が変わった画面を
# 捨ててしまう危険がある。

# pHashが何ビットまで違えば「同じ画面」とみなすか(64ビット中)
_DUPLICATE_HASH_DISTANCE = 2

# 重複でも必ず保存する種別。manualはユーザーの明示操作、initialはセッションの先頭
_ALWAYS_STORED_KINDS = frozenset({"manual", "initial"})


@dataclass(slots=True)
class _StoredFrame:
    """直前に保存したフレーム。次のフレームが重複かどうかの判定に使う。"""

    session_id: str
    frame: np.ndarray
    phash: str
    event_id: int


class OcrUnavailable(RuntimeError):
    """A missing/loading OCR provider is not a successful empty recognition."""


class EventProcessor:
    def __init__(
        self,
        db: Database,
        storage: Storage,
        ocr: OcrProvider,
        embeddings: EmbeddingProvider,
        queue_size: int = 8,
    ):
        self.db = db
        self.storage = storage
        self.ocr = ocr
        self.embeddings = embeddings
        self.queue: LatestQueue[ProcessingItem] = LatestQueue(queue_size, self._drop_item)
        self.status = ProcessorStatus()
        self.chrome_tracker = ChromeLineTracker()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._index_lock = threading.Lock()
        self._store_lock = threading.Lock()
        # 重複判定は直前に保存した1枚と比べる。同期処理とワーカーの双方を
        # _store_lockで直列化し、同じ手動要求を重ねても一度だけ保存する。
        self._previous_stored: _StoredFrame | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.queue.reopen()
        self.status.state = "running"
        self._thread = threading.Thread(target=self._run, daemon=True, name="event-processor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.queue.close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)
        elif self._thread is None:
            while (item := self.queue.get(timeout=0)) is not None:
                self._drop_item(item, "Storage processor stopped before saving the frame")
        self.status.state = "stopping" if self._thread and self._thread.is_alive() else "stopped"

    def submit(self, session_id: str, candidate: FrameCandidate) -> bool:
        accepted_without_drop = self.queue.put(ProcessingItem(session_id, candidate))
        self.status.queue_depth = len(self.queue)
        self.status.queue_drops = self.queue.drops
        return accepted_without_drop

    @staticmethod
    def _drop_item(item: ProcessingItem, reason: str) -> None:
        if item.candidate.manual_request:
            item.candidate.manual_request.finish(error=reason)

    def manual_capture_status(self, request_id: str) -> dict | None:
        row = self.db.fetchone("SELECT id FROM screen_event WHERE manual_request_id=?", (request_id,))
        if row is None:
            return None
        return {"request_id": request_id, "state": "saved", "event_id": int(row["id"]), "error": None}

    def _duplicate_event_id(self, item: ProcessingItem, phash: str) -> int | None:
        """直前に保存した画面と同じなら、そのevent_idを返す。違えばNone。"""
        if item.candidate.event_kind in _ALWAYS_STORED_KINDS:
            return None
        previous = self._previous_stored
        if previous is None or previous.session_id != item.session_id:
            return None
        if previous.frame.shape != item.candidate.frame.shape:
            return None
        if hash_distance(previous.phash, phash) > _DUPLICATE_HASH_DISTANCE:
            return None
        settings = self.storage.settings
        if any(
            region_changed_fraction(
                previous.frame, item.candidate.frame, settings.pixel_delta, Region(**value).normalized()
            )
            >= settings.watch_change_threshold
            for value in settings.watch_regions
        ):
            return None
        # 差分から除外する領域(時計など)は、変化検出と同じ扱いにする。
        # 「無視してよい」と設定された場所の変化で新しい画面とみなすのは筋が通らない
        changed = changed_fraction(
            previous.frame,
            item.candidate.frame,
            settings.pixel_delta,
            [Region(**value).normalized() for value in settings.ignore_regions],
        )
        if changed > settings.duplicate_change_threshold:
            return None
        return previous.event_id

    def store_one(self, item: ProcessingItem) -> int | None:
        with self._store_lock:
            try:
                return self._store_one(item)
            except Exception as exc:
                if item.candidate.manual_request:
                    item.candidate.manual_request.finish(error=str(exc))
                raise

    def _store_one(self, item: ProcessingItem) -> int | None:
        """フレームを保存してscreen_eventを作成する。OCR・埋め込みは行わない(processed_atはNULLのまま)。

        キャプチャ中はこのメソッドのみを呼び、索引処理(_index_event)はBackgroundIndexerが
        記録停止後にまとめて実行することでCPU負荷を分離する。

        直前に保存した画面と同じ内容だった場合は保存せず ``None`` を返す。
        """
        request = item.candidate.manual_request
        if request:
            existing = self.manual_capture_status(request.request_id)
            if existing:
                request.finish(event_id=existing["event_id"])
                return existing["event_id"]
        phash = perceptual_hash(item.candidate.frame)
        duplicate_of = self._duplicate_event_id(item, phash)
        if duplicate_of is not None:
            # 画像は捨てるが、その画面が映っていた時間は失わないよう終了時刻だけ伸ばす
            with self.db.transaction() as connection:
                updated = connection.execute(
                    "UPDATE screen_event SET ended_at=MAX(ended_at, ?) WHERE id=?",
                    (iso(item.candidate.ended_at), duplicate_of),
                ).rowcount
            if updated:
                self.status.skipped_duplicates += 1
                return None
            self._previous_stored = None
        if not self.storage.has_capacity():
            self.status.state = "blocked-storage"
            error = "Storage limit or minimum free-space threshold reached"
            if request:
                request.finish(error=error)
            raise RuntimeError(error)
        self.status.state = "running"
        stored = self.storage.save_frame(item.candidate.frame, item.session_id, item.candidate.ended_at)
        try:
            with self.db.transaction() as connection:
                previous = connection.execute(
                    "SELECT id FROM screen_event WHERE session_id=? ORDER BY id DESC LIMIT 1",
                    (item.session_id,),
                ).fetchone()
                event_id = int(
                    connection.execute(
                        """INSERT INTO screen_event(
                        session_id, started_at, ended_at, frame_path, thumbnail_path, phash,
                        change_score, width, height, event_kind, previous_event_id, created_at,
                        captured_at, manual_request_id, indexing_state, frame_bytes, thumbnail_bytes
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'pending',?,?)""",
                        (
                            item.session_id,
                            iso(item.candidate.started_at),
                            iso(item.candidate.ended_at),
                            str(stored.frame_path),
                            str(stored.thumbnail_path),
                            phash,
                            item.candidate.change_score,
                            stored.width,
                            stored.height,
                            item.candidate.event_kind,
                            previous["id"] if previous else None,
                            iso(datetime.now(UTC)),
                            iso(item.candidate.captured_at or item.candidate.ended_at),
                            request.request_id if request else None,
                            stored.frame_bytes,
                            stored.thumbnail_bytes,
                        ),
                    ).lastrowid
                )
                connection.execute(
                    """INSERT INTO evidence(
                           event_id, kind, path, mime_type, sha256, metadata_json, created_at)
                       VALUES(?, 'frame', ?, 'image/webp', ?, '{}', ?)""",
                    (event_id, str(stored.frame_path), stored.sha256, iso(datetime.now(UTC))),
                )
        except Exception:
            self.storage.remove_paths([stored.frame_path, stored.thumbnail_path])
            raise
        # 次のフレームとの比較用に、いま保存した1枚を控える
        self._previous_stored = _StoredFrame(
            session_id=item.session_id,
            frame=item.candidate.frame.copy(),
            phash=phash,
            event_id=event_id,
        )
        if self.status.last_saved_session_id != item.session_id:
            self.status.session_saved_count = 0
        self.status.saved_count += 1
        self.status.processed = self.status.saved_count
        self.status.session_saved_count += 1
        self.status.last_saved_at = iso(datetime.now(UTC))
        self.status.last_processed_at = self.status.last_saved_at
        self.status.last_saved_kind = item.candidate.event_kind
        self.status.last_saved_event_id = event_id
        self.status.last_saved_session_id = item.session_id
        self.status.last_error = None
        if request:
            request.finish(event_id=event_id)
        return event_id

    def process_one(self, item: ProcessingItem) -> int | None:
        """保存 + 索引を同期的に行う(テスト・後方互換用)。通常の記録中はstore_oneのみが使われる。

        直前と同じ画面で保存を見送った場合は ``None`` を返す。
        """
        event_id = self.store_one(item)
        if event_id is None:
            return None
        try:
            self._index_event(event_id, item.candidate.frame, item.session_id)
        except Exception as exc:
            LOGGER.exception("Indexing failed for event %s", event_id)
            self.status.last_error = str(exc)
            self.status.failures += 1
            self._mark_index_failure(event_id, exc)
        return event_id

    def reprocess_one(self, event_id: int) -> int:
        """Retry OCR and embedding generation for an already stored event."""
        row = self.db.fetchone("SELECT frame_path, session_id FROM screen_event WHERE id=?", (event_id,))
        if not row:
            raise LookupError("Event not found")
        self.db.execute(
            "UPDATE screen_event SET indexing_state='pending', indexing_error=NULL, "
            "index_attempts=0, index_retry_at=NULL WHERE id=?",
            (event_id,),
        )
        try:
            with self.storage.open_image(row["frame_path"]) as image:
                frame = to_bgr(image)
            self._index_event(event_id, frame, str(row["session_id"]))
        except Exception as exc:
            LOGGER.exception("Reindexing failed for event %s", event_id)
            self.status.last_error = str(exc)
            self.status.failures += 1
            self._mark_index_failure(event_id, exc)
            raise
        return event_id

    def _mark_index_failure(self, event_id: int, error: Exception) -> None:
        self.db.execute(
            "UPDATE screen_event SET indexing_state=?,indexing_error=?,index_retry_at=NULL WHERE id=?",
            ("blocked" if isinstance(error, OcrUnavailable) else "failed", str(error), event_id),
        )

    def _clean_ocr_text(self, session_id: str, ocr_result: OcrResult, height: int) -> str:
        """タスクバー等のUIノイズを除いたOCRテキストを組み立てる。

        画面をフル解像度で取り込んでいるため、本文と一緒に常時表示のUI(タスクバー・
        時計・メニューバー)もOCRされる。これらは全フレームに共通して現れるので、
        残したまま埋め込みを作るとどのフレーム同士も似た向きのベクトルになり、
        意味検索で内容の違いを区別しにくくなる。全文検索でも毎回ヒットしてしまう。
        """
        # 1つのOcrLineに複数行が入っている場合(改行を含むテキストを返すプロバイダ)も
        # 行単位で判定できるよう、改行で分割してから扱う
        lines = [
            {"text": part, "polygon": line.polygon}
            for line in ocr_result.lines
            for part in line.text.splitlines()
        ]
        if not lines and ocr_result.text:
            # 行データを返さないOCRプロバイダでも、本文が空にならないようにする
            # (座標が無いので位置による判定は効かず、パターン判定のみになる)
            lines = [{"text": value, "polygon": []} for value in ocr_result.text.splitlines()]
        repeated = self.chrome_tracker.observe(session_id, {line_key(str(line["text"])) for line in lines})
        return "\n".join(clean_ocr_lines(lines, height, repeated))

    def _index_event(self, event_id: int, frame: np.ndarray, session_id: str) -> None:
        # OCRバックエンドは同時呼び出しを保証しないため、録画処理と手動再処理を直列化する。
        with self._index_lock:
            if not self.ocr.available:
                raise OcrUnavailable(self.ocr.reason or "OCR is unavailable")
            # 段組みを考慮した読み順に直してから正規化する。OCRエンジンによって
            # 行の返し方が違うため、どのエンジンでも同じ順序になるようここで揃える
            ocr_result = normalize_ocr_result(order_ocr_result(self.ocr.recognize(frame)))
            clean_text = self._clean_ocr_text(session_id, ocr_result, int(frame.shape[0]))
            # 埋め込みは洗浄後のテキストから作る(生テキストではない)
            vector = self.embeddings.encode_document(clean_text) if clean_text else None
        metadata = json.dumps(
            {
                "provider": self.ocr.name,
                "lines": [
                    {"text": line.text, "confidence": line.confidence, "polygon": line.polygon}
                    for line in ocr_result.lines
                ],
            },
            ensure_ascii=False,
        )
        processed_at = iso(datetime.now(UTC))
        with self.db.transaction() as connection:
            connection.execute(
                """
                UPDATE screen_event SET
                    ocr_text=?, ocr_text_clean=?, ocr_confidence=?,
                    embedding=?, embedding_dim=?, processed_at=?, indexing_state='ready',
                    indexing_error=NULL, index_attempts=0, index_retry_at=NULL
                WHERE id=?
                """,
                (
                    # 生テキストは証跡とLIKEでの厳密検索のために残し、
                    # 全文検索の索引には洗浄後のテキストを使う
                    ocr_result.text,
                    clean_text,
                    ocr_result.confidence,
                    vector.astype(np.float32).tobytes() if vector is not None else None,
                    int(vector.size) if vector is not None else None,
                    processed_at,
                    event_id,
                ),
            )
            connection.execute(
                """INSERT INTO evidence(event_id, kind, text, mime_type, metadata_json, created_at)
                   VALUES(?, 'ocr', ?, 'text/plain', ?, ?)""",
                (event_id, ocr_result.text, metadata, processed_at),
            )

    def _run(self) -> None:
        while not self._stop.is_set() or len(self.queue):
            item = self.queue.get(timeout=0.5)
            if item is None:
                continue
            try:
                self.store_one(item)
            except Exception as exc:
                LOGGER.exception("Event storage failed")
                self.status.failures += 1
                self.status.last_error = str(exc)
                if item.candidate.manual_request:
                    item.candidate.manual_request.finish(error=str(exc))
            self.status.queue_depth = len(self.queue)
            self.status.queue_drops = self.queue.drops


@dataclass(slots=True)
class IndexerStatus:
    state: str = "idle"  # idle | indexing | paused-capture
    pending_count: int = 0
    indexed_total: int = 0
    failures: int = 0
    last_error: str | None = None
    last_indexed_at: str | None = None
    failed_count: int = 0
    blocked_count: int = 0


class BackgroundIndexer:
    """未索引(processed_at IS NULL)のイベントを古い順にOCR・埋め込み(索引)処理する。

    CPUでのOCRは1枚あたり数十秒かかる重い処理のため、キャプチャ中に同時実行すると
    CPUを占有しプレビュー・手動キャプチャの応答が遅れる。既定では記録中は待機し、
    停止している間だけ処理する。ただしOCRが別プロセスワーカー(GPU)で動いている
    構成ではメインプロセスの負荷が小さいため、allow_during_captureで記録中の
    処理継続を許可できる(準リアルタイムに検索へ反映される)。
    """

    def __init__(
        self,
        db: Database,
        processor: EventProcessor,
        is_capture_active: Callable[[], bool],
        allow_during_capture: Callable[[], bool] | None = None,
        poll_interval: float = 2.0,
        idle_interval: float = 10.0,
        max_attempts: int = 3,
    ):
        self.db = db
        self.processor = processor
        self.is_capture_active = is_capture_active
        # OCRが別プロセスワーカー(GPU)で動いている等、メインプロセスのCPUを
        # 奪わない構成のときだけ記録中の索引を許可するための判定フック
        self.allow_during_capture = allow_during_capture or (lambda: False)
        self.poll_interval = poll_interval
        self.idle_interval = idle_interval
        self.max_attempts = max_attempts
        self.status = IndexerStatus()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="background-indexer")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=15)

    def notify(self) -> None:
        """記録停止時などに呼び出し、待機中のポーリング間隔を待たずに再確認させる。"""
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.is_capture_active() and not self.allow_during_capture():
                self.status.state = "paused-capture"
                self._refresh_pending_count()
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            if not self.processor.ocr.available:
                self.status.state = "blocked-ocr"
                self.status.last_error = self.processor.ocr.reason or "OCR is unavailable"
                self.db.execute(
                    "UPDATE screen_event SET indexing_state='blocked',indexing_error=? "
                    "WHERE indexing_state IN ('pending','retry')",
                    (self.status.last_error,),
                )
                self._refresh_pending_count()
                self._wake.wait(self.poll_interval)
                self._wake.clear()
                continue
            row = self._next_pending()
            if row is None:
                self._refresh_pending_count()
                self.status.state = "retry-wait" if self.status.pending_count else "idle"
                self._wake.wait(self.poll_interval if self.status.pending_count else self.idle_interval)
                self._wake.clear()
                continue
            self.status.state = "indexing"
            # 1枚の処理には数十秒かかることがあるため、処理前に残数を更新しておく
            # (起動直後のバックログ処理で「残り0枚」と表示されるのを防ぐ)
            self._refresh_pending_count()
            self._process_row(row)

    def _next_pending(self):
        return self.db.fetchone(
            "SELECT id, frame_path, session_id, index_attempts FROM screen_event "
            "WHERE indexing_state IN ('pending','blocked','retry') "
            "AND (index_retry_at IS NULL OR index_retry_at <= ?) "
            "ORDER BY COALESCE(captured_at,started_at),id LIMIT 1",
            (iso(datetime.now(UTC)),),
        )

    def _refresh_pending_count(self) -> None:
        rows = self.db.fetchall(
            "SELECT indexing_state,COUNT(*) AS count FROM screen_event GROUP BY indexing_state"
        )
        counts = {row["indexing_state"]: int(row["count"]) for row in rows}
        self.status.pending_count = sum(counts.get(key, 0) for key in ("pending", "blocked", "retry"))
        self.status.failed_count = counts.get("failed", 0)
        self.status.blocked_count = counts.get("blocked", 0)

    def _process_row(self, row) -> None:
        event_id = int(row["id"])
        try:
            with self.processor.storage.open_image(row["frame_path"]) as image:
                frame = to_bgr(image)
            self.processor._index_event(event_id, frame, str(row["session_id"]))
            self.status.indexed_total += 1
            self.status.last_indexed_at = iso(datetime.now(UTC))
            self.status.last_error = None
        except OcrUnavailable as exc:
            self.processor._mark_index_failure(event_id, exc)
            self.status.state = "blocked-ocr"
            self.status.last_error = str(exc)
        except Exception as exc:
            LOGGER.exception("Background indexing failed for event %s", event_id)
            self.status.failures += 1
            self.status.last_error = str(exc)
            current = self.db.fetchone("SELECT index_attempts FROM screen_event WHERE id=?", (event_id,))
            attempts = int(current["index_attempts"]) + 1 if current else 1
            failed = attempts >= self.max_attempts
            delay = min(60.0, max(0.05, self.poll_interval) * 2 ** (attempts - 1))
            retry_at = None if failed else iso(datetime.now(UTC) + timedelta(seconds=delay))
            self.db.execute(
                "UPDATE screen_event SET indexing_state=?,indexing_error=?,index_attempts=?,"
                "index_retry_at=? WHERE id=?",
                ("failed" if failed else "retry", str(exc), attempts, retry_at, event_id),
            )
        finally:
            self._refresh_pending_count()


class RetentionService:
    def __init__(self, db: Database, storage: Storage):
        self.db = db
        self.storage = storage

    def cleanup(self, now: datetime | None = None) -> int:
        cutoff = iso(self.storage.retention_cutoff(now))
        with self.db.transaction() as connection:
            rows = connection.execute(
                """SELECT id, frame_path, thumbnail_path FROM screen_event
                   WHERE ended_at < ? AND retained=0
                   AND id NOT IN (SELECT event_id FROM context_pack_item)""",
                (cutoff,),
            ).fetchall()
            if not rows:
                return 0
            connection.executemany("DELETE FROM screen_event WHERE id=?", [(row["id"],) for row in rows])
        self.storage.remove_paths(
            [path for row in rows for path in (row["frame_path"], row["thumbnail_path"])]
        )
        return len(rows)
