from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from visual_memory.ai import (
    AsyncEmbeddingProvider,
    AsyncOcrProvider,
    OcrResult,
    SubprocessOcrProvider,
    YomiTokuOcrProvider,
    build_ocr_provider,
)


def test_yomitoku_provider_returns_sorted_lines_and_coordinates(monkeypatch):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    yomitoku = ModuleType("yomitoku")

    class Detector:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "cuda"

        def __call__(self, frame):
            return SimpleNamespace(points=[[[20, 20], [30, 20], [30, 30], [20, 30]]]), None

    class Recognizer:
        def __init__(self, **kwargs):
            assert kwargs == {"model_name": "parseq-tiny", "device": "cuda"}

        def __call__(self, frame, points):
            return (
                SimpleNamespace(
                    contents=["lower", "upper"],
                    scores=[0.8, 0.9],
                    points=[
                        [[20, 20], [30, 20], [30, 30], [20, 30]],
                        [[1, 1], [10, 1], [10, 10], [1, 10]],
                    ],
                ),
                None,
            )

    yomitoku.TextDetector = Detector
    yomitoku.TextRecognizer = Recognizer
    monkeypatch.setitem(sys.modules, "yomitoku", yomitoku)

    provider = YomiTokuOcrProvider(lite=True, device="auto")
    result = provider.recognize(np.zeros((40, 40, 3), dtype=np.uint8))

    assert provider.name == "yomitoku-lite-cuda"
    assert result.text == "upper\nlower"
    assert result.confidence == 0.8500000000000001
    assert result.lines[0].polygon[0] == [1.0, 1.0]


def test_unknown_ocr_provider_fails_closed():
    provider = build_ocr_provider("not-real")
    assert not provider.available
    assert "Unknown OCR provider" in provider.reason


def test_subprocess_worker_reports_initialization_failure_without_protocol_corruption():
    with pytest.raises(RuntimeError, match="Unknown OCR provider"):
        SubprocessOcrProvider(sys.executable, "not-real", "cpu")


def test_async_providers_do_not_block_startup_and_become_ready():
    class Ocr:
        name = "ready-ocr"
        available = True
        reason = None

        def recognize(self, frame):
            return OcrResult("ready", 1.0, [])

    class Embeddings:
        name = "ready-embedding"
        available = True
        reason = None
        dimension = 2

        def encode_document(self, text):
            return np.asarray([1, 0], dtype=np.float32)

        def encode_query(self, text):
            return np.asarray([0, 1], dtype=np.float32)

    ocr = AsyncOcrProvider(Ocr)
    embeddings = AsyncEmbeddingProvider(Embeddings)
    assert ocr.state == "loading"
    assert embeddings.encode_query("available exact search") is None

    ocr.start()
    embeddings.start()
    assert ocr.recognize(np.zeros((2, 2, 3), dtype=np.uint8)).text == "ready"
    vector = embeddings.encode_document("document")

    assert ocr.state == "ready"
    assert embeddings.state == "ready"
    assert vector.tolist() == [1.0, 0.0]
