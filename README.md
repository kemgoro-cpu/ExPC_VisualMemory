# External PC Visual Memory

キャプチャーボードから取得した別PCの画面を、ローカルで検索可能な履歴へ変換します。生の画面履歴はAIへ公開されません。検索UIで選んで作成したコンテキスト文書だけをMCPから取得できます。

## 次版に向けた改善（ソースへ反映済み）

- 手動キャプチャは画像・サムネイル・DB保存完了を確認して通知。保存中の結果照会と同じ要求IDでの再試行に対応
- OCRの待機・利用不可・再試行・失敗を区別し、検索の時刻更新と短い複数語の扱いを修正
- 同じOCRでも図表が異なる画面を残し、文書の設定変更・同じIDでの再生成・画像拡大・削除を追加
- 文書生成中の共有停止を保証。再生成時も共有期限・停止状態を維持
- 文書と元画像の長期保持を通常履歴から分け、容量内訳を表示

## 0.2.2の主な変更

- 個人利用構成に合わせ、墨消しのUI・APIとBitLocker確認処理を削除
- 過去の文書に保存済みの墨消し情報は、再生成時の互換性維持のため引き続き適用

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
- 録画単位のタイムライン、表示中の画面選択、ドラッグ複数選択
- 画像とスクラブ済みOCRを埋め込んだ単一HTML／PDFコンテキスト文書
- 24時間を既定とするMCP利用期限と共有停止
- 作成済みコンテキスト文書だけを公開するstdio MCPサーバー
- ドラッグ並べ替え、逆順、時系列順に対応した文書バスケット
- 画面画像・重要OCRを持つ単一HTML/PDF。PDFへのOCR全文付録は任意選択（既定OFF）
- 通常履歴は30日保持。文書と文書に使った元画像は長期保持し、共有停止・期限切れ・文書削除では元画像を消さない
- 容量上限、空き容量監視、USB切断時の自動再接続

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

文書作成が完了すると直ちに共有され、既定24時間後にMCPからの取得が止まります。期限はローカル保存の期限ではありません。文書詳細から設定を変更して再生成しても、共有期限の延長や共有停止の解除は行いません。

画像の重複除去で図表が欠けた既存文書や、OCR全文付録が必要な既存PDFは、文書詳細で設定を変更して再生成してください。元画像が残っている必要があり、既存文書を自動で書き換えることはありません。

## データとセキュリティ

既定の保存先は`%LOCALAPPDATA%\ExternalPCVisualMemory`です。APIは`127.0.0.1`にだけバインドし、ランダムトークンとCSRFトークンで保護します。

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

出力名は`outputs\external-pc-visual-memory-lite-0.2.2`と`outputs\external-pc-visual-memory-full-0.2.2`です。各配布物にSHA-256、初回起動ガイド、MCP設定例を同梱します。

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
