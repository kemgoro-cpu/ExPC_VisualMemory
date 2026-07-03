from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from visual_memory.ai import SubprocessOcrProvider


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("python_executable", type=Path)
    parser.add_argument("provider", choices=("paddle", "yomitoku", "yomitoku-lite"))
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    frame = cv2.imread(str(args.image.resolve()), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit("Unable to read smoke-test image")
    provider = SubprocessOcrProvider(str(args.python_executable.resolve()), args.provider, "cuda")
    try:
        result = provider.recognize(frame)
        if not result.text.strip():
            raise SystemExit("OCR worker returned no text")
        print(provider.name)
        print(result.text)
    finally:
        provider.close()


if __name__ == "__main__":
    main()
