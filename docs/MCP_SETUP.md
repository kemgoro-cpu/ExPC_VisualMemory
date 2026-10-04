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

作成完了と同時に共有が始まり、既定24時間後に取得できなくなります。ローカルの文書と元画像は期限切れや共有停止後も残ります。文書を再生成しても期限延長や共有再開は行いません。生成処理と共有停止が重なっても、停止操作を優先してMCPからの取得を止めます。

PDFの全文OCR付録は文書ごとの設定です（既定OFF）。AIにOCR全文も渡したい場合は、UIの文書詳細から全文付録をONにして再生成してください。`get_context_pack`の構造化OCR情報と、`get_context_document`が返すPDFの内容は同一とは限りません。

保存先を変更して起動している場合は、MCPサーバーにも同じ場所を渡します。

```json
{
  "command": "C:\\path\\to\\visual-memory-mcp.exe",
  "args": ["--data-dir", "D:\\VisualMemoryData"]
}
```
