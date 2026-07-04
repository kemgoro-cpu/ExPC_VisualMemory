from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from io import TextIOWrapper

import anyio
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents

from .config import load_settings
from .db import Database
from .packs import ContextPackService

mcp = FastMCP(
    "External PC Visual Memory",
    instructions=(
        "Only user-created context documents are available. Raw history, legacy drafts, revoked packs, "
        "and expired packs are intentionally inaccessible."
    ),
)


def _service() -> ContextPackService:
    settings = load_settings()
    database = Database(settings.database_path)
    database.initialize()
    return ContextPackService(database, settings)


# artifact_path やmanifest_sha256などのローカルファイルシステム内部情報はAIに公開しない
_PUBLIC_PACK_FIELDS = {
    "id",
    "title",
    "note",
    "query",
    "status",
    "item_count",
    "deduplicate_overlaps",
    "created_at",
    "approved_at",
    "expires_at",
}


def _public_pack(pack: dict) -> dict:
    return {key: value for key, value in pack.items() if key in _PUBLIC_PACK_FIELDS}


@mcp.tool()
def list_context_packs(search: str = "") -> str:
    """List user-created context documents that are currently available."""
    packs = _service().list(include_drafts=False, search=search)
    return json.dumps([_public_pack(pack) for pack in packs], ensure_ascii=False, indent=2)


@mcp.tool()
def search_context_packs(query: str) -> str:
    """Search titles, notes, and originating queries within available context documents."""
    if not query.strip():
        raise ValueError("query must not be empty")
    packs = _service().list(include_drafts=False, search=query.strip())
    return json.dumps([_public_pack(pack) for pack in packs], ensure_ascii=False, indent=2)


@mcp.tool()
def get_context_pack(pack_id: str) -> str:
    """Return the manifest and cleaned OCR text for one available context document."""
    service = _service()
    pack = service.get_approved(pack_id)
    manifest = service.artifact_dir(pack) / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError("Context document manifest is missing")
    return manifest.read_text(encoding="utf-8")


@mcp.tool()
def get_context_document(pack_id: str, document_format: str = "pdf") -> EmbeddedResource:
    """Return one context document as a single PDF or self-contained HTML file."""
    service = _service()
    path = service.get_approved_document(pack_id, document_format)
    uri = f"visual-memory://context-packs/{pack_id}/{path.name}"
    if path.suffix == ".html":
        resource = TextResourceContents(
            uri=uri,
            mimeType="text/html",
            text=path.read_text(encoding="utf-8"),
        )
    else:
        resource = BlobResourceContents(
            uri=uri,
            mimeType="application/pdf",
            blob=base64.b64encode(path.read_bytes()).decode("ascii"),
        )
    return EmbeddedResource(type="resource", resource=resource)


async def _run_stdio() -> None:
    # The SDK's UTF-8 wrappers close their underlying stream during teardown. Use duplicate
    # handles so a frozen Windows executable keeps the bootloader's original stdio alive.
    stdin_binary = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    stdout_binary = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    stdin = anyio.wrap_file(TextIOWrapper(stdin_binary, encoding="utf-8", errors="replace"))
    stdout = anyio.wrap_file(TextIOWrapper(stdout_binary, encoding="utf-8"))
    async with stdio_server(stdin=stdin, stdout=stdout) as (read_stream, write_stream):
        await mcp._mcp_server.run(
            read_stream,
            write_stream,
            mcp._mcp_server.create_initialization_options(),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="visual-memory-mcp")
    parser.add_argument("--data-dir", help="Override the local Visual Memory data directory")
    parser.add_argument("--mcp", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.data_dir:
        os.environ["VISUAL_MEMORY_DATA_DIR"] = args.data_dir
    anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
