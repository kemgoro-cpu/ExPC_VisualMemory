from __future__ import annotations

import argparse
import logging
import os
import threading
import webbrowser

import uvicorn

from . import __version__
from .api import create_app
from .config import load_settings
from .mcp_server import main as mcp_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visual-memory", description="External PC Visual Memory")
    parser.add_argument("--data-dir", help="Override the local data directory")
    parser.add_argument("--host", default=None, help="Bind host; defaults to 127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="Bind port; defaults to 8765")
    parser.add_argument("--no-open", action="store_true", help="Do not open the UI in the default browser")
    parser.add_argument("--mcp", action="store_true", help="Run the context-document stdio MCP server")
    parser.add_argument("--log-level", default="info", choices=("debug", "info", "warning", "error"))
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.mcp:
        if args.data_dir:
            os.environ["VISUAL_MEMORY_DATA_DIR"] = args.data_dir
        mcp_main([])
        return
    settings = load_settings(args.data_dir)
    if args.host:
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit("Refusing to bind outside localhost")
        settings.host = args.host
    if args.port:
        settings.port = args.port
    app = create_app(settings)
    url = f"http://{settings.host}:{settings.port}/?token={settings.auth_token}"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Visual Memory: {url}")
    print(f"Data: {settings.data_dir}")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
