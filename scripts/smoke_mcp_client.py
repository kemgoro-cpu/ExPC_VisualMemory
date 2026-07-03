from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def smoke(command: str, data_dir: str) -> None:
    dedicated_server = Path(command).stem.lower().endswith("-mcp")
    args = ["--data-dir", data_dir] if dedicated_server else ["--mcp", "--data-dir", data_dir]
    parameters = StdioServerParameters(command=command, args=args)
    async with (
        stdio_client(parameters) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        names = sorted(tool.name for tool in tools.tools)
        expected = [
            "get_context_document",
            "get_context_pack",
            "list_context_packs",
            "search_context_packs",
        ]
        print(names)
        if names != expected:
            raise RuntimeError(f"Unexpected MCP tool surface: {names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command")
    parser.add_argument("data_dir")
    args = parser.parse_args()
    asyncio.run(smoke(args.command, args.data_dir))


if __name__ == "__main__":
    main()
