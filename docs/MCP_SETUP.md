# MCP設定

配布物の`visual-memory-mcp\visual-memory-mcp.exe`を、MCPクライアントのstdioサーバーとして登録します。

```json
{
  "mcpServers": {
    "visual-memory": {
      "command": "C:\\path\\to\\visual-memory-mcp\\visual-memory-mcp.exe",
      "args": []
    }
  }
}
```

利用可能なツールは次の4つです。

- `list_context_packs`
- `search_context_packs`
- `get_context_pack`
- `get_context_document`

MCPから取得できるのは、UIで作成され、期限内かつ共有停止されていないコンテキスト文書だけです。生の画面履歴へアクセスするツールはありません。

保存先を変更して起動している場合は、MCPサーバーにも同じ場所を渡します。

```json
{
  "command": "C:\\path\\to\\visual-memory-mcp.exe",
  "args": ["--data-dir", "D:\\VisualMemoryData"]
}
```
