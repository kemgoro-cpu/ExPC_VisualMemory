"""Download local OCR and embedding models before building an offline package."""

import argparse
import shutil
from pathlib import Path

from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, help="Create a model bundle for PyInstaller")
    args = parser.parse_args()
    PaddleOCR(
        lang="japan",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    embedding = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    if args.output_dir:
        output = args.output_dir.resolve()
        paddlex_output = output / "paddlex"
        paddlex_output.mkdir(parents=True, exist_ok=True)
        cache = Path.home() / ".paddlex" / "official_models"
        for name in ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"):
            shutil.copytree(cache / name, paddlex_output / name, dirs_exist_ok=True)
        embedding.save(str(output / "multilingual-e5-small"))
        print(f"Offline model bundle: {output}")
    print("OCR and embedding models are ready.")


if __name__ == "__main__":
    main()
