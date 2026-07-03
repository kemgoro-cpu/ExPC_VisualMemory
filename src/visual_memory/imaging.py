from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def to_pil(frame: np.ndarray) -> Image.Image:
    if frame.ndim == 2:
        return Image.fromarray(frame)
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def small_gray(frame: np.ndarray, width: int = 320, height: int = 180) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True, slots=True)
class Region:
    x: float
    y: float
    width: float
    height: float

    def normalized(self) -> Region:
        x = min(1.0, max(0.0, self.x))
        y = min(1.0, max(0.0, self.y))
        width = min(1.0 - x, max(0.0, self.width))
        height = min(1.0 - y, max(0.0, self.height))
        return Region(x=x, y=y, width=width, height=height)


def _region_bounds(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
    item = region.normalized()
    left = round(item.x * width)
    top = round(item.y * height)
    right = round((item.x + item.width) * width)
    bottom = round((item.y + item.height) * height)
    return left, top, right, bottom


def changed_fraction(
    previous: np.ndarray,
    current: np.ndarray,
    pixel_delta: int,
    ignore_regions: Iterable[Region] = (),
) -> float:
    before = small_gray(previous)
    after = small_gray(current)
    delta = cv2.absdiff(before, after)
    mask = np.ones(delta.shape, dtype=bool)
    for region in ignore_regions:
        left, top, right, bottom = _region_bounds(region, delta.shape[1], delta.shape[0])
        mask[top:bottom, left:right] = False
    eligible = int(np.count_nonzero(mask))
    if not eligible:
        return 0.0
    return float(np.count_nonzero((delta >= pixel_delta) & mask) / eligible)


def region_changed_fraction(
    previous: np.ndarray, current: np.ndarray, pixel_delta: int, region: Region
) -> float:
    height, width = previous.shape[:2]
    left, top, right, bottom = _region_bounds(region, width, height)
    if right <= left or bottom <= top:
        return 0.0
    return changed_fraction(previous[top:bottom, left:right], current[top:bottom, left:right], pixel_delta)


def perceptual_hash(frame: np.ndarray) -> str:
    gray = small_gray(frame, 32, 32)
    dct = cv2.dct(np.float32(gray))[:8, :8]
    median = float(np.median(dct[1:, :]))
    bits = (dct > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hash_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_webp(frame: np.ndarray, quality: int = 90) -> bytes:
    buffer = io.BytesIO()
    to_pil(frame).save(buffer, "WEBP", quality=quality, method=4)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class Redaction:
    x: float
    y: float
    width: float
    height: float

    def normalized(self) -> Redaction:
        x = min(1.0, max(0.0, self.x))
        y = min(1.0, max(0.0, self.y))
        width = min(1.0 - x, max(0.0, self.width))
        height = min(1.0 - y, max(0.0, self.height))
        return Redaction(x=x, y=y, width=width, height=height)


def apply_redactions(image: Image.Image, redactions: Iterable[Redaction]) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for raw in redactions:
        item = raw.normalized()
        left = round(item.x * output.width)
        top = round(item.y * output.height)
        right = round((item.x + item.width) * output.width)
        bottom = round((item.y + item.height) * output.height)
        if right > left and bottom > top:
            draw.rectangle((left, top, right, bottom), fill="black")
    return output
