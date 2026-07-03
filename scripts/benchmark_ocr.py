from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

from visual_memory.ai import build_ocr_provider

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PROVIDERS = ("paddle", "yomitoku", "yomitoku-lite")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def character_error_rate(expected: str, actual: str) -> float:
    left, right = normalize_text(expected), normalize_text(actual)
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        current = [row]
        for column, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1] / len(left)


class GpuMemorySampler:
    def __init__(self) -> None:
        self.samples: list[int] = []
        self.total_mib: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> tuple[int, int] | None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            used, total = output.splitlines()[0].split(",")
            return int(used.strip()), int(total.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def start(self) -> None:
        initial = self._read()
        if initial:
            self.samples.append(initial[0])
            self.total_mib = initial[1]

        def sample() -> None:
            while not self._stop.wait(0.1):
                value = self._read()
                if value:
                    self.samples.append(value[0])
                    self.total_mib = value[1]

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def load_cases(dataset: Path) -> list[tuple[Path, Path, str]]:
    cases: list[tuple[Path, Path, str]] = []
    for image in sorted(path for path in dataset.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
        truth = image.with_suffix(".txt")
        if not truth.exists():
            continue
        relative = image.relative_to(dataset)
        category = relative.parts[0] if len(relative.parts) > 1 else "uncategorized"
        cases.append((image, truth, category))
    if not cases:
        raise SystemExit("No image/.txt pairs found in the benchmark dataset")
    return cases


def run_provider(provider_name: str, dataset: Path) -> dict[str, Any]:
    cases = load_cases(dataset)
    sampler = GpuMemorySampler()
    baseline = sampler._read()
    sampler.start()
    load_started = time.perf_counter()
    provider = build_ocr_provider(provider_name, device="cuda")
    model_load_seconds = time.perf_counter() - load_started
    if not provider.available:
        sampler.stop()
        return {"provider": provider_name, "error": provider.reason, "cases": []}

    results: list[dict[str, Any]] = []
    for image_path, truth_path, category in cases:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            results.append({"image": str(image_path), "category": category, "error": "read failed"})
            continue
        expected = truth_path.read_text(encoding="utf-8")
        started = time.perf_counter()
        try:
            recognized = provider.recognize(frame)
            latency = time.perf_counter() - started
            results.append(
                {
                    "image": str(image_path.relative_to(dataset)),
                    "category": category,
                    "expected": expected,
                    "actual": recognized.text,
                    "cer": character_error_rate(expected, recognized.text),
                    "latency_seconds": latency,
                    "confidence": recognized.confidence,
                }
            )
        except Exception as exc:
            results.append({"image": str(image_path), "category": category, "error": str(exc)})
    sampler.stop()

    valid = [item for item in results if "error" not in item]
    latencies = [float(item["latency_seconds"]) for item in valid]
    by_category: dict[str, list[float]] = defaultdict(list)
    for item in valid:
        by_category[item["category"]].append(float(item["cer"]))
    category_cer = {key: statistics.mean(values) for key, values in sorted(by_category.items())}
    peak = max(sampler.samples) if sampler.samples else None
    baseline_used = baseline[0] if baseline else None
    total = sampler.total_mib or (baseline[1] if baseline else None)
    return {
        "provider": provider_name,
        "runtime_name": provider.name,
        "model_load_seconds": model_load_seconds,
        "case_count": len(results),
        "error_count": len(results) - len(valid),
        "mean_cer": statistics.mean(float(item["cer"]) for item in valid) if valid else 1.0,
        "macro_category_cer": statistics.mean(category_cer.values()) if category_cer else 1.0,
        "category_cer": category_cer,
        "p50_latency_seconds": percentile(latencies, 0.50),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "gpu_baseline_mib": baseline_used,
        "gpu_peak_mib": peak,
        "gpu_peak_delta_mib": peak - baseline_used
        if peak is not None and baseline_used is not None
        else None,
        "gpu_total_mib": total,
        "cases": results,
    }


def choose_winner(results: list[dict[str, Any]]) -> str | None:
    eligible = [
        item
        for item in results
        if not item.get("error")
        and item.get("error_count") == 0
        and item.get("p95_latency_seconds", 99) <= 5.0
        and (
            item.get("gpu_peak_mib") is None
            or item.get("gpu_total_mib") is None
            or item["gpu_peak_mib"] <= item["gpu_total_mib"] * 0.92
        )
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: item["macro_category_cer"])["provider"]


def write_report(results: list[dict[str, Any]], output: Path) -> None:
    winner = choose_winner(results)
    payload = {
        "winner": winner,
        "selection_rules": {"p95_seconds": 5.0, "max_vram_ratio": 0.92},
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "ocr-benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# OCR GPU Benchmark", "", f"Recommended provider: **{winner or 'none'}**", ""]
    lines += [
        "| Provider | Macro CER | Mean CER | P95 latency | Peak VRAM | Errors |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        if item.get("error"):
            lines.append(f"| {item['provider']} | - | - | - | - | {item['error']} |")
            continue
        peak = f"{item['gpu_peak_mib']} MiB" if item.get("gpu_peak_mib") is not None else "n/a"
        lines.append(
            f"| {item['provider']} | {item['macro_category_cer']:.3f} | "
            f"{item['mean_cer']:.3f} | {item['p95_latency_seconds']:.2f}s | "
            f"{peak} | {item['error_count']} |"
        )
    lines += ["", "Lower CER is better. Selection excludes providers above 5s P95 or 92% VRAM."]
    (output / "ocr-benchmark.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PaddleOCR and YomiToku on captured frames")
    parser.add_argument("dataset", type=Path, help="Folders of image files with same-name .txt truth files")
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=list(PROVIDERS))
    parser.add_argument("--output", type=Path, default=Path("work/ocr-benchmark"))
    parser.add_argument("--paddle-python", type=Path)
    parser.add_argument("--yomitoku-python", type=Path)
    parser.add_argument("--child-provider", choices=PROVIDERS)
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()

    if args.child_provider:
        result = run_provider(args.child_provider, args.dataset.resolve())
        args.child_output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for provider in args.providers:
        child_output = output / f"{provider}.json"
        configured_python = args.paddle_python if provider == "paddle" else args.yomitoku_python
        interpreter = configured_python.resolve() if configured_python else Path(sys.executable)
        subprocess.run(
            [
                str(interpreter),
                str(Path(__file__).resolve()),
                str(args.dataset.resolve()),
                "--child-provider",
                provider,
                "--child-output",
                str(child_output),
            ],
            check=True,
        )
        results.append(json.loads(child_output.read_text(encoding="utf-8")))
    write_report(results, output)
    print(output / "ocr-benchmark.md")


if __name__ == "__main__":
    main()
