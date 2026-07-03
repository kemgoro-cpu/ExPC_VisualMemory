from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a frozen Visual Memory build")
    parser.add_argument("executable", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("work/frozen-smoke"))
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--ocr-worker-python", type=Path)
    parser.add_argument(
        "--ocr-provider", choices=("disabled", "paddle", "yomitoku", "yomitoku-lite")
    )
    parser.add_argument("--expect-device", help="Require this DirectShow device in the frozen app")
    args = parser.parse_args()

    executable = args.executable.resolve()
    data_dir = args.data_dir.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = data_dir / "stdout.log"
    stderr_path = data_dir / "stderr.log"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    environment = os.environ.copy()
    if args.ocr_provider:
        environment["VISUAL_MEMORY_OCR_PROVIDER"] = args.ocr_provider
    if args.ocr_worker_python:
        environment["VISUAL_MEMORY_OCR_WORKER_PYTHON"] = str(args.ocr_worker_python.resolve())
        environment.setdefault("VISUAL_MEMORY_OCR_PROVIDER", "paddle")
        environment["VISUAL_MEMORY_OCR_DEVICE"] = "cuda"

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [
                str(executable),
                "--data-dir",
                str(data_dir),
                "--port",
                str(args.port),
                "--no-open",
                "--log-level",
                "info",
            ],
            stdout=stdout,
            stderr=stderr,
            env=environment,
        )
        try:
            deadline = time.monotonic() + args.timeout
            token_path = data_dir / ".auth-token"
            while time.monotonic() < deadline:
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                        healthy = response.status == 200
                except OSError:
                    healthy = False
                if healthy and token_path.exists():
                    token = token_path.read_text(encoding="utf-8").strip()
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{args.port}/api/status",
                        headers={"X-Visual-Memory-Token": token},
                    )
                    try:
                        # The status probe also checks BitLocker and can take a
                        # few seconds on machines without management privileges.
                        with opener.open(request, timeout=30) as response:
                            status = json.load(response)
                        if args.expect_device:
                            device_request = urllib.request.Request(
                                f"http://127.0.0.1:{args.port}/api/devices",
                                headers={"X-Visual-Memory-Token": token},
                            )
                            with opener.open(device_request, timeout=30) as response:
                                devices = json.load(response)["devices"]
                            if args.expect_device not in devices:
                                raise SystemExit(
                                    f"Expected DirectShow device {args.expect_device!r}; got {devices!r}"
                                )
                            status["device_probe"] = {"devices": devices}
                        print(json.dumps(status, ensure_ascii=False, indent=2))
                        return
                    except OSError:
                        pass
                if process.poll() is not None:
                    break
                time.sleep(1)
            raise SystemExit("Frozen app did not become ready.\n" + stderr_path.read_text(errors="replace"))
        finally:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )


if __name__ == "__main__":
    main()
