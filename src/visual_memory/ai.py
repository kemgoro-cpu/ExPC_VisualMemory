from __future__ import annotations

import base64
import json
import logging
import queue
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class OcrLine:
    text: str
    confidence: float
    polygon: list[list[float]]


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    confidence: float | None
    lines: list[OcrLine]


class OcrProvider(Protocol):
    name: str
    available: bool
    reason: str | None

    def recognize(self, frame: np.ndarray) -> OcrResult: ...


class EmbeddingProvider(Protocol):
    name: str
    available: bool
    dimension: int | None
    reason: str | None

    def encode_document(self, text: str) -> np.ndarray | None: ...

    def encode_query(self, text: str) -> np.ndarray | None: ...


class DisabledOcr:
    name = "disabled"
    available = False

    def __init__(self, reason: str):
        self.reason = reason

    def recognize(self, frame: np.ndarray) -> OcrResult:
        return OcrResult(text="", confidence=None, lines=[])


class DisabledEmbedding:
    name = "disabled"
    available = False
    dimension = None

    def __init__(self, reason: str):
        self.reason = reason

    def encode_document(self, text: str) -> None:
        return None

    def encode_query(self, text: str) -> None:
        return None


class AsyncOcrProvider:
    """Load an OCR backend without delaying the local web UI."""

    def __init__(self, factory: Callable[[], OcrProvider]):
        self._factory = factory
        self._provider: OcrProvider | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "loading"
        self._error: str | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def name(self) -> str:
        return self._provider.name if self._provider else "initializing"

    @property
    def available(self) -> bool:
        return bool(self._provider and self._provider.available)

    @property
    def reason(self) -> str | None:
        if self._provider:
            return self._provider.reason
        return self._error or "OCR model is loading"

    @property
    def fallback_reason(self) -> str | None:
        """GPUワーカー起動失敗などでCPUへフォールバックした場合の理由(通常はNone)。"""
        return getattr(self._provider, "fallback_reason", None)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._ready.is_set():
            return
        self._thread = threading.Thread(target=self._load, daemon=True, name="ocr-model-loader")
        self._thread.start()

    def _load(self) -> None:
        try:
            self._provider = self._factory()
            self._state = "ready" if self._provider.available else "disabled"
        except Exception as exc:
            LOGGER.exception("OCR model initialization failed")
            self._error = str(exc)
            self._provider = DisabledOcr(f"Unable to initialize OCR: {exc}")
            self._state = "error"
        finally:
            self._ready.set()

    def recognize(self, frame: np.ndarray) -> OcrResult:
        self.start()
        self._ready.wait()
        assert self._provider is not None
        return self._provider.recognize(frame)

    def close(self) -> None:
        provider = self._provider
        close = getattr(provider, "close", None)
        if close:
            close()


class AsyncEmbeddingProvider:
    """Load embeddings in the background while exact search stays available."""

    def __init__(self, factory: Callable[[], EmbeddingProvider]):
        self._factory = factory
        self._provider: EmbeddingProvider | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "loading"
        self._error: str | None = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def name(self) -> str:
        return self._provider.name if self._provider else "initializing"

    @property
    def available(self) -> bool:
        return bool(self._provider and self._provider.available)

    @property
    def dimension(self) -> int | None:
        return self._provider.dimension if self._provider else None

    @property
    def reason(self) -> str | None:
        if self._provider:
            return self._provider.reason
        return self._error or "Embedding model is loading"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._ready.is_set():
            return
        self._thread = threading.Thread(
            target=self._load,
            daemon=True,
            name="embedding-model-loader",
        )
        self._thread.start()

    def _load(self) -> None:
        try:
            self._provider = self._factory()
            self._state = "ready" if self._provider.available else "disabled"
        except Exception as exc:
            LOGGER.exception("Embedding model initialization failed")
            self._error = str(exc)
            self._provider = DisabledEmbedding(f"Unable to initialize embeddings: {exc}")
            self._state = "error"
        finally:
            self._ready.set()

    def encode_document(self, text: str) -> np.ndarray | None:
        self.start()
        self._ready.wait()
        assert self._provider is not None
        return self._provider.encode_document(text)

    def encode_query(self, text: str) -> np.ndarray | None:
        if not self._ready.is_set() or not self._provider:
            return None
        return self._provider.encode_query(text)


class PaddleOcrProvider:
    name = "paddleocr"
    available = True
    reason = None

    def __init__(
        self,
        language: str = "japan",
        detection_model_dir: str | None = None,
        recognition_model_dir: str | None = None,
        device: str = "auto",
    ):
        from paddleocr import PaddleOCR

        model_options = {}
        active_device = device
        if device == "auto":
            try:
                import paddle

                active_device = "cuda" if paddle.device.is_compiled_with_cuda() else "cpu"
            except Exception:
                active_device = "cpu"
        if detection_model_dir:
            model_options["text_detection_model_name"] = "PP-OCRv6_medium_det"
            model_options["text_detection_model_dir"] = detection_model_dir
        if recognition_model_dir:
            model_options["text_recognition_model_name"] = "PP-OCRv6_medium_rec"
            model_options["text_recognition_model_dir"] = recognition_model_dir
        model_options["device"] = "gpu:0" if active_device == "cuda" else active_device
        # OCRの認識時間そのものは変わらないが、CPUスレッドを絞ることでWebサーバーや
        # キャプチャスレッドにCPUを譲りやすくする(索引処理中の応答性のため)
        model_options["cpu_threads"] = 4
        try:
            self.engine = PaddleOCR(
                lang=None if model_options else language,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                **model_options,
            )
            self._api = "predict"
        except TypeError:
            self.engine = PaddleOCR(lang=language, use_angle_cls=True, show_log=False)
            self._api = "legacy"
        self.name = f"paddleocr-{active_device}"

    def recognize(self, frame: np.ndarray) -> OcrResult:
        raw = self.engine.predict(frame) if self._api == "predict" else self.engine.ocr(frame, cls=True)
        lines = self._parse(raw)
        confidence = float(np.mean([line.confidence for line in lines])) if lines else None
        return OcrResult(
            text="\n".join(line.text for line in lines if line.text.strip()),
            confidence=confidence,
            lines=lines,
        )

    def _parse(self, raw: Any) -> list[OcrLine]:
        lines: list[OcrLine] = []
        if not raw:
            return lines
        for result in raw:
            payload = self._payload(result)
            if payload:
                texts = payload.get("rec_texts", [])
                scores = payload.get("rec_scores", [])
                polygons = payload.get("rec_polys", payload.get("dt_polys", []))
                for index, text in enumerate(texts):
                    score = float(scores[index]) if index < len(scores) else 0.0
                    polygon = (
                        np.asarray(polygons[index]).astype(float).tolist() if index < len(polygons) else []
                    )
                    lines.append(OcrLine(str(text), score, polygon))
                continue
            for item in result or []:
                if not item or len(item) < 2:
                    continue
                polygon, recognized = item[0], item[1]
                text = str(recognized[0])
                score = float(recognized[1])
                lines.append(OcrLine(text, score, np.asarray(polygon).astype(float).tolist()))
        return lines

    @staticmethod
    def _payload(result: Any) -> dict[str, Any] | None:
        value = getattr(result, "json", None)
        if callable(value):
            value = value()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        if isinstance(value, dict):
            return value.get("res", value)
        if isinstance(result, dict):
            return result.get("res", result)
        return None


class YomiTokuOcrProvider:
    available = True
    reason = None

    def __init__(self, *, lite: bool = False, device: str = "auto"):
        import torch
        from yomitoku import TextDetector, TextRecognizer

        active_device = device
        if device == "auto":
            active_device = "cuda" if torch.cuda.is_available() else "cpu"
        if active_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("YomiToku CUDA was requested but torch.cuda is unavailable")
        recognizer_name = "parseq-tiny" if lite else "parseq-large-v4_1"
        self.detector = TextDetector(device=active_device)
        self.recognizer = TextRecognizer(model_name=recognizer_name, device=active_device)
        self.name = f"yomitoku-{'lite' if lite else 'normal'}-{active_device}"

    def recognize(self, frame: np.ndarray) -> OcrResult:
        detection, _ = self.detector(frame)
        recognition, _ = self.recognizer(frame, detection.points)
        lines = [
            OcrLine(str(text), float(score), np.asarray(polygon).astype(float).tolist())
            for text, score, polygon in zip(
                recognition.contents,
                recognition.scores,
                recognition.points,
                strict=False,
            )
            if str(text).strip()
        ]
        lines.sort(
            key=lambda line: (
                min((point[1] for point in line.polygon), default=0),
                min((point[0] for point in line.polygon), default=0),
            )
        )
        confidence = float(np.mean([line.confidence for line in lines])) if lines else None
        return OcrResult(text="\n".join(line.text for line in lines), confidence=confidence, lines=lines)


class SubprocessOcrProvider:
    available = True
    reason = None

    def __init__(self, python_executable: str, provider_name: str, device: str):
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [
                python_executable,
                "-m",
                "visual_memory.ocr_worker",
                "--provider",
                provider_name,
                "--device",
                device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

        def read_responses() -> None:
            assert self._process.stdout is not None
            for line in self._process.stdout:
                marker = line.find(b"@@VMOCR@@")
                if marker < 0:
                    LOGGER.debug("Ignoring OCR worker output: %s", line.decode(errors="replace").rstrip())
                    continue
                try:
                    payload = json.loads(line[marker + len(b"@@VMOCR@@") :].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    LOGGER.warning("Ignoring malformed OCR worker protocol line")
                    continue
                self._responses.put(payload)

        self._reader = threading.Thread(target=read_responses, daemon=True, name="ocr-worker-reader")
        self._reader.start()
        ready = self._receive(timeout=180)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "OCR worker failed to initialize"))
        self.name = f"worker:{ready['name']}"

    def _receive(self, timeout: float) -> dict[str, Any]:
        try:
            return self._responses.get(timeout=timeout)
        except queue.Empty as exc:
            code = self._process.poll()
            raise RuntimeError(f"OCR worker timed out (exit={code})") from exc

    def recognize(self, frame: np.ndarray) -> OcrResult:
        import cv2

        success, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not success:
            raise RuntimeError("Unable to encode frame for OCR worker")
        request_id = str(uuid.uuid4())
        payload = {
            "id": request_id,
            "op": "recognize",
            "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
        with self._lock:
            if self._process.poll() is not None or self._process.stdin is None:
                raise RuntimeError("OCR worker is not running")
            wire = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
            self._process.stdin.write(wire)
            self._process.stdin.flush()
            while True:
                response = self._receive(timeout=60)
                if response.get("id") == request_id:
                    break
        if response.get("error"):
            raise RuntimeError(response["error"])
        lines = [
            OcrLine(
                text=str(item["text"]),
                confidence=float(item["confidence"]),
                polygon=item.get("polygon", []),
            )
            for item in response.get("lines", [])
        ]
        return OcrResult(
            text=str(response.get("text", "")),
            confidence=response.get("confidence"),
            lines=lines,
        )

    def close(self) -> None:
        if getattr(self, "_process", None) is None or self._process.poll() is not None:
            return
        if self._process.stdin:
            try:
                self._process.stdin.write(b'{"op":"shutdown"}\n')
                self._process.stdin.flush()
            except OSError:
                pass
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.terminate()


class SentenceTransformerEmbedding:
    available = True
    reason = None

    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        # デフォルトの論理コア数(このマシンで14)をそのまま使うとWebサーバーや
        # キャプチャスレッドからCPUを奪ってしまうため、索引処理中も譲れるよう絞る
        torch.set_num_threads(4)
        self.name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = int(self.model.get_embedding_dimension())
        else:  # sentence-transformers < 5.6
            self.dimension = int(self.model.get_sentence_embedding_dimension())

    def _encode(self, value: str) -> np.ndarray:
        result = self.model.encode(value, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(result, dtype=np.float32)

    def encode_document(self, text: str) -> np.ndarray | None:
        text = text.strip()
        return self._encode(f"passage: {text}") if text else None

    def encode_query(self, text: str) -> np.ndarray | None:
        text = text.strip()
        return self._encode(f"query: {text}") if text else None


def _build_inprocess_provider(
    provider_name: str,
    detection_model_dir: str | None,
    recognition_model_dir: str | None,
    device: str,
) -> OcrProvider:
    if provider_name in {"yomitoku", "yomitoku-lite"}:
        return YomiTokuOcrProvider(lite=provider_name.endswith("-lite"), device=device)
    if provider_name != "paddle":
        return DisabledOcr(f"Unknown OCR provider: {provider_name}")
    return PaddleOcrProvider(
        detection_model_dir=detection_model_dir,
        recognition_model_dir=recognition_model_dir,
        device=device,
    )


def build_ocr_provider(
    provider_name: str = "paddle",
    detection_model_dir: str | None = None,
    recognition_model_dir: str | None = None,
    device: str = "auto",
    worker_python: str | None = None,
) -> OcrProvider:
    if provider_name == "disabled":
        return DisabledOcr("OCR is disabled in config")
    if worker_python:
        try:
            return SubprocessOcrProvider(worker_python, provider_name, device)
        except Exception as exc:
            # GPUワーカーが起動できなくてもOCRを完全停止させず、インプロセスCPUで継続する。
            # deviceを"cpu"に固定するのは、ワーカー分離の理由だったCUDA DLL競合を
            # メインプロセスへ持ち込まないため
            LOGGER.warning("OCR worker failed to start; falling back to in-process CPU OCR: %s", exc)
            try:
                fallback = _build_inprocess_provider(
                    provider_name, detection_model_dir, recognition_model_dir, "cpu"
                )
            except Exception as inner:
                LOGGER.warning("%s OCR unavailable: %s", provider_name, inner)
                return DisabledOcr(f"Unable to initialize {provider_name} OCR: {inner}")
            fallback.fallback_reason = f"OCR worker failed to start: {exc}"
            return fallback
    try:
        return _build_inprocess_provider(
            provider_name, detection_model_dir, recognition_model_dir, device
        )
    except Exception as exc:
        LOGGER.warning("%s OCR unavailable: %s", provider_name, exc)
        return DisabledOcr(f"Unable to initialize {provider_name} OCR: {exc}")


def build_embedding_provider(model_name: str) -> EmbeddingProvider:
    try:
        return SentenceTransformerEmbedding(model_name)
    except Exception as exc:
        LOGGER.warning("Embedding model unavailable: %s", exc)
        return DisabledEmbedding(f"Install the ai extra to enable semantic search: {exc}")
