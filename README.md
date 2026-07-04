# External PC Visual Memory

キャプチャーボードから取得した別PCの画面を、ローカルで検索可能な履歴へ変換します。生の画面履歴はAIへ公開されません。検索UIで選んで作成したコンテキスト文書だけをMCPから取得できます。文書は作成時点で利用可能になり、作成後の墨消しにも追従して再生成されます。

## 0.2.1の主な変更

- FFmpegの警告出力による停止を防ぎ、検索の短語・時刻境界・埋め込みキャッシュを修正
- 承認期限、認証トークン、履歴削除、設定破損時の復旧を堅牢化
- 保存済みイベントのOCR・検索再処理と、失敗した文書の再生成操作を追加
- イベントカードのキーボード選択と、非表示タブでの状態ポーリング停止に対応

## 0.2.0の主な変更

- ローカルUIを先に起動し、OCR・意味検索モデルをバックグラウンドで準備
- 保存容量を毎回全走査せず、差分カウンターと1時間ごとの整合確認で管理
- コンテキスト文書をステージング領域で完成させてから原子的に公開
- 画面履歴の段階読み込み、入力デバイス記憶、操作ボタンの状態制御を追加
- 狭いウィンドウでも横スクロールしないレスポンシブ表示
- Lite版とFull版を明示的に分ける再現可能なリリーススクリプト

## 現在の機能

- Windows DirectShow対応キャプチャーデバイスをFFmpegで取得
- 画面変化、静止、定期チェックポイントによる代表フレーム化
- PaddleOCRによる日本語・英語OCRと多言語E5による意味検索
- SQLite FTS5とローカル埋め込みを組み合わせた検索
- 録画単位のタイムライン、録画ごとの全選択、ドラッグ複数選択、画像の墨消し
- 画像とスクラブ済みOCRを埋め込んだ単一HTML／PDFコンテキスト文書
- 24時間を既定とするMCP利用期限と共有停止
- 作成済みコンテキスト文書だけを公開するstdio MCPサーバー
- ドラッグ並べ替え、逆順、時系列順に対応した文書バスケット
- 1画面1ページ、重要OCR、ノイズ除去済み全文OCR付録を持つ単一HTML/PDF
- 30日保持、容量上限、空き容量監視、USB切断時の自動再接続

## 配布版を使う

Windows配布物を展開し、`visual-memory\visual-memory.exe`を実行します。PythonとFFmpegを別途インストールする必要はありません。初回操作は[docs/FIRST_RUN.md](docs/FIRST_RUN.md)、MCP登録は[docs/MCP_SETUP.md](docs/MCP_SETUP.md)を参照してください。

| プロファイル | 内容 | 用途 |
|---|---|---|
| Lite | UI、キャプチャー、画像保存、MCP。AIモデルなし | 小容量の動作確認 |
| Full | PaddleOCR、多言語E5、ローカルモデル同梱 | 完全オフライン運用 |

UIはすぐに開きます。Full版ではモデル準備が完了するまで状態が表示され、記録開始ボタンは安全のため無効になります。

## 開発環境のセットアップ

Python 3.12とFFmpegが必要です。PowerShellで次を実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ai,dev]"
visual-memory
```

起動時にローカルUIが開きます。初回のOCR／埋め込みモデル取得後は、コアアプリからクラウドへ通信しません。完全オフライン運用では、先にモデルを取得してからネットワークを切断してください。

AI依存を入れずUIとキャプチャーだけ確認する場合：

```powershell
python -m pip install -e ".[dev]"
visual-memory
```

この場合も画像は保存されますが、OCRと意味検索は停止状態としてUIに表示されます。

## MCP設定

MCPクライアントから次のコマンドをstdioサーバーとして登録します。

```text
visual-memory-mcp
```

PyInstaller版では、MCP専用の配布ディレクトリにあるEXEを使用します。

```text
visual-memory-mcp.exe
```

公開ツールは次の4つだけです。

- `list_context_packs`
- `search_context_packs`
- `get_context_pack`
- `get_context_document`

共有停止済み、期限切れ文書と、生の画面履歴を取得するMCPツールは存在しません。`get_context_document`は単一HTMLまたはPDFだけを返します。

## データとセキュリティ

既定の保存先は`%LOCALAPPDATA%\ExternalPCVisualMemory`です。APIは`127.0.0.1`にだけバインドし、ランダムトークンとCSRFトークンで保護します。保存先ドライブのBitLockerが確認できない場合はUIに警告します。

画面画像は非常に機密性が高いため、Windowsユーザーを共有せず、BitLockerを有効にしてください。SSD上の削除を「安全消去」とは扱いません。

## テスト

```powershell
pytest
```

## Windowsパッケージ

モデルとPyInstallerを導入した環境で、Lite版とFull版を同時に作成できます。既存の同名出力がある場合は上書きせず停止します。

```powershell
python scripts/prefetch_models.py --output-dir work\model-bundle
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1 -Profile Both
```

出力名は`outputs\external-pc-visual-memory-lite-0.2.1`と`outputs\external-pc-visual-memory-full-0.2.1`です。各配布物にSHA-256、初回起動ガイド、MCP設定例を同梱します。

## GPU OCR（非商用プロファイル）

PaddleOCRとYomiTokuはCUDA DLLが競合するため、GPU OCRを本体とは別の永続ワーカープロセスで動かします。最初に環境を作成します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_gpu.ps1
```

通常は実測で選ばれたPaddleOCRを起動します。YomiTokuを試す場合は`-Provider yomitoku`または`yomitoku-lite`を指定します。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_gpu.ps1 -Provider paddle
```

OCR比較用データは、画像と同名のUTF-8 `.txt`正解ファイルをカテゴリ別フォルダへ置きます。

```text
benchmark/
  code/editor.png
  code/editor.txt
  powerpoint/slide.png
  powerpoint/slide.txt
```

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ocr.py benchmark `
  --paddle-python .venv-paddle-gpu\Scripts\python.exe `
  --yomitoku-python .venv-yomi-gpu\Scripts\python.exe `
  --output work\ocr-benchmark
```

配布フォルダでは次のラッパーを使用できます。

```powershell
powershell -ExecutionPolicy Bypass -File benchmark_gpu.ps1 -Dataset benchmark
```

選定条件はカテゴリ平均CER最小、P95が5秒以内、ピークVRAMが総量の92%以下です。合成スモークデータではPaddleOCRが首位でしたが、最終決定には実キャプチャ画像を使ってください。

`dist\visual-memory\`と`dist\visual-memory-mcp\`が`onedir`配布物です。Webアプリ側にはビルド時に検出したFFmpegが同梱されます。
