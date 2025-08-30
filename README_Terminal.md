# ターミナル版チャット UI（Textual）

本ドキュメントは、`terminal_app_strands.py` を使ったターミナル UI 版の起動・設定方法を日本語でまとめたものです。複数 LLM（Gemini / OpenAI / Anthropic）と、複数 MCP サーバ（SSE / stdio）を統合して利用できます。

## できること
- Markdown でAI応答を表示（コードブロックやリストを綺麗に描画）
- アバター（ASCIIアート）と同期した発話表示
- 複数 MCP サーバからのツールを Strands Agent で統合
- 接続済み MCP 数を「connected / configured」で表示

## 動作要件
- Python 3.10+ 推奨
- パッケージは `requirements.txt` からインストール

```bash
# 仮想環境の利用を推奨
python -m venv venv
source venv/bin/activate  # Windows は venv\Scripts\activate

# 依存関係のインストール（pip または uv など）
pip install -r requirements.txt
# もしくは
uv pip install -r requirements.txt
```

主な依存:
- Textual（ターミナル UI）
- Rich（Markdown レンダリング）
- LangChain（各 LLM ドライバ）
- strands-agents / strands-agents-tools（Strands Agent 本体と MCP ツール）
- mcp（stdio / sse クライアント）

## 設定（.env）
設定は `.env` に記述します。テンプレートは `/.env.example` を参照してください。

最低限、以下を設定してください（例は同梱の `.env` より）。

```dotenv
# 使用する LLM（google-genai | openai | anthropic）
LLM_PROVIDER=google-genai

# Google AI Studio の API キー（LLM_PROVIDER=google-genai の場合）
GEMINI_API_KEY=...
# 使うモデル
MODEL_NAME=gemini-2.5-flash

# MCP を使う場合のスイッチ
MCP_ENABLED=True

# 複数 MCP サーバの設定（JSON 文字列）
MCP_SERVERS_JSON='{
  "taivily":  {"transport":"sse",   "url":"http://localhost:5001/sse"},
  "firecrawl":{"transport":"sse",   "url":"http://localhost:5002/sse"},
  "weather":  {"transport":"sse",   "url":"http://localhost:5003/sse"},
  "time":     {"transport":"sse",   "url":"http://localhost:5004/sse"}
}'
```

- `transport` は `sse` もしくは `stdio` を指定できます。
  - `sse` の場合は `url` が必須
  - `stdio` の場合は `command`（必要なら `args`）を指定
- `settings.py` 側で URL のよくある誤記（`http:/` → `http://` など）を軽微に補正します。

UI関連の主な可変値:
- `AVATAR_NAME`, `AVATAR_FULL_NAME`（アバター名）
- `AVATAR_IMAGE_IDLE`, `AVATAR_IMAGE_TALK`（画像が無い場合は ASCII にフォールバック）
- `TYPEWRITER_DELAY_MS`, `MOUTH_ANIMATION_INTERVAL_MS`（表示/口パク速度）

## 起動方法
```bash
# ターミナル UI を起動
python terminal_app_strands.py
```
起動時、上部に以下のようなステータスを表示します。
- 例: `Shoebill Communicator オンライン | LLM: google-genai | MCP: ✓ (1 connected / 4 configured)`
  - configured: .env に設定された MCP サーバ数
  - connected: 実際に接続できたサーバ数（ツールが取得できたもの）

## 使い方
- 画面下部の入力欄にメッセージを入力して Enter
- 右側のアバターが発話モードになり、左側ログに Markdown 描画で応答が流れます
- ショートカット: `q` または `Ctrl+C` で終了

## LLM 切替
- `.env` の `LLM_PROVIDER` を切り替えます
  - `google-genai`: `GEMINI_API_KEY` と `MODEL_NAME` を利用
  - `openai`: `OPENAI_API_KEY` と `OPENAI_MODEL`（任意、デフォルト `gpt-4o`）
  - `anthropic`: `ANTHROPIC_API_KEY` と `ANTHROPIC_MODEL`（任意）

## MCP について
- 複数サーバのツールを起動時に統合し、Strands Agent に渡します
- SSE/stdio どちらも可。エラーになったサーバはスキップし、使えるものだけで動作します
- 接続時の詳細ログは UI には出さず、内部で抑制しています

SSE と stdio の例:
```dotenv
MCP_SERVERS_JSON='{
  "web":  {"transport":"sse",   "url":"http://localhost:6001/sse", "headers": {"Authorization":"Bearer ..."}},
  "local":{"transport":"stdio", "command":"/usr/local/bin/my-mcp", "args":["--flag"]}
}'
```

## トラブルシュート
- MCP が `✓ (1 connected / 4 configured)` のように一部しか接続されない
  - 対象サーバの SSE エンドポイントが生きているか（`curl http://localhost:PORT/sse`）
  - URL パス（`/sse` か `/mcp` か）と認証ヘッダの有無
  - stdio の場合は `command` の実行パスと権限
- 画面が真っ黒／ログが出ない
  - ほとんどの初期化ログは抑制済み。`.env` の `DEBUG_MODE=True` にしてから `settings.py` の標準出力を確認
- Ascii アートが表示されない
  - `static/images/` に `AVATAR_IMAGE_IDLE`, `AVATAR_IMAGE_TALK` が無い場合は ASCII に自動フォールバックします

## 主要ファイル
- `terminal_app_strands.py`：ターミナル UI 本体（Strands + MCP + LLM 統合）
- `settings.py`：`.env` 読み込みと検証、MCP サーバ設定のパース
- `.env`：環境変数による設定

## ライセンス
- 本リポジトリの `LICENSE` を参照してください

