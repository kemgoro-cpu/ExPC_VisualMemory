from __future__ import annotations

import argparse
import base64
import contextlib
import json
import sys

import cv2
import numpy as np

from .ai import build_ocr_provider


def send(payload: dict) -> None:
    wire = ("@@VMOCR@@" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    sys.__stdout__.buffer.write(wire)
    sys.__stdout__.buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            provider = build_ocr_provider(args.provider, device=args.device)
        if not provider.available:
            send({"ready": False, "error": provider.reason})
            return
        send({"ready": True, "name": provider.name})
    except Exception as exc:
        send({"ready": False, "error": str(exc)})
        return

    request: dict = {}
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if request.get("op") == "shutdown":
                break
            request_id = request.get("id")
            image = np.frombuffer(base64.b64decode(request["image"]), dtype=np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Unable to decode input image")
            with contextlib.redirect_stdout(sys.stderr):
                result = provider.recognize(frame)
            send(
                {
                    "id": request_id,
                    "text": result.text,
                    "confidence": result.confidence,
                    "lines": [
                        {
                            "text": line.text,
                            "confidence": line.confidence,
                            "polygon": line.polygon,
                        }
                        for line in result.lines
                    ],
                }
            )
        except Exception as exc:
            send({"id": request.get("id"), "error": str(exc)})


if __name__ == "__main__":
    main()
