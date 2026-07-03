from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from visual_memory.ai import PaddleOcrProvider, SentenceTransformerEmbedding


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the bundled OCR and embedding models")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    detector = bundle / "paddlex" / "PP-OCRv6_medium_det"
    recognizer = bundle / "paddlex" / "PP-OCRv6_medium_rec"
    embedding_dir = bundle / "multilingual-e5-small"

    image = Image.new("RGB", (1280, 180), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\meiryo.ttc", 48)
    expected = "第3四半期の利益率は42%に改善"
    draw.text((35, 45), expected, fill="black", font=font)

    ocr = PaddleOcrProvider(
        detection_model_dir=str(detector),
        recognition_model_dir=str(recognizer),
    )
    result = ocr.recognize(np.asarray(image))

    embeddings = SentenceTransformerEmbedding(str(embedding_dir))
    query = embeddings.encode_query("利益率が改善した四半期")
    document = embeddings.encode_document(expected)
    similarity = float(np.dot(query, document))

    if expected not in result.text:
        raise SystemExit(f"Bundled OCR verification failed: {result.text!r}")
    if embeddings.dimension != 384 or similarity < 0.7:
        raise SystemExit(
            f"Bundled embedding verification failed: dimension={embeddings.dimension}, "
            f"similarity={similarity:.4f}"
        )

    print(f"OCR: {result.text}")
    print(f"OCR confidence: {result.confidence:.4f}")
    print(f"Embedding: dimension={embeddings.dimension}, similarity={similarity:.4f}")


if __name__ == "__main__":
    main()
