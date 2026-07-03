# Build after running: python scripts/prefetch_models.py
import importlib.util
import os
import shutil
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

datas = [
    ("src/visual_memory/templates", "visual_memory/templates"),
    ("src/visual_memory/static", "visual_memory/static"),
]
build_profile = os.environ.get("VISUAL_MEMORY_BUILD_PROFILE", "full").lower()
if build_profile not in {"lite", "full"}:
    raise ValueError("VISUAL_MEMORY_BUILD_PROFILE must be 'lite' or 'full'")
include_ai = build_profile == "full"
excludes = [] if include_ai else [
    "paddle",
    "paddleocr",
    "paddlex",
    "sentence_transformers",
    "sklearn",
    "torch",
    "transformers",
    "yomitoku",
]
model_bundle = Path(os.environ.get("VISUAL_MEMORY_MODEL_BUNDLE", "assets/models"))
if include_ai and model_bundle.exists():
    datas.append((str(model_bundle), "models"))
binaries = []
ffmpeg = shutil.which("ffmpeg")
if ffmpeg:
    binaries.append((ffmpeg, "."))
hiddenimports = ["mcp.server.fastmcp"]
for package in ("fastapi", "uvicorn"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden
for package in (("paddleocr", "sentence_transformers") if include_ai else ()):
    if importlib.util.find_spec(package):
        package_datas, package_binaries, package_hidden = collect_all(package)
        datas += package_datas
        binaries += package_binaries
        hiddenimports += package_hidden
if include_ai and importlib.util.find_spec("paddlex"):
    # PaddleOCR resolves its pipeline by reading paddlex/configs/pipelines/OCR.yaml.
    # PyInstaller discovers the Python modules but not these runtime YAML files.
    datas += collect_data_files("paddlex", includes=["configs/**"])
    # PaddleX checks these distribution names through importlib.metadata before
    # constructing the OCR processors, so their dist-info must survive freezing.
    for distribution in (
        "paddlex",
        "imagesize",
        "opencv-contrib-python",
        "pyclipper",
        "pypdfium2",
        "python-bidi",
        "shapely",
    ):
        datas += copy_metadata(distribution)
if include_ai and importlib.util.find_spec("paddle"):
    binaries += collect_dynamic_libs("paddle")

a = Analysis(
    ["scripts/pyinstaller_entry.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="visual-memory", console=True)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True, name="visual-memory")
