# ターミナル版チャット UI（Textual）

本ドキュメントは、`terminal_app_strands.py` / `terminal_app_fish.py` を使ったターミナル UI 版の起動・設定方法を日本語でまとめたものです。複数 LLM（Gemini / OpenAI / Anthropic / Ollama / OpenAI互換API）と、複数 MCP サーバ（SSE / stdio）を統合して利用できます。
また、アバター表示として PNG 表示（PICモード）と Boids による魚群表示（FISHモード）に対応し、画像アバターの ASCII フォールバック、タイプ音（任意）やスラッシュコマンドにも対応しています。

## できること
- Markdown でAI応答を表示（コードブロックやリストを綺麗に描画）
- アバター（ASCIIアート）と同期した発話表示
- 複数 MCP サーバからのツールを Strands Agent で統合
- 接続済み MCP 数を「connected / configured」で表示
- スラッシュコマンド: `/clear`（画面クリア）、`/mcp`（利用可能ツール一覧）
- タイプ音（numpy + pygame があれば有効化）

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
- strands-agents / strands-agents-tools（Strands Agent 本体と MCP ツール）
- mcp（stdio / sse クライアント）
（任意機能）
- Pillow + rich[image] または rich-pixels（画像アバター描画）
- ascii-magic（画像が無い/使えない場合の ASCII 変換）
- numpy + pygame（タイプ音）

## 設定（.env）
設定は `.env` に記述します。テンプレートは `/.env.example` を参照してください。

最低限、以下を設定してください（例は同梱の `.env` より）。

```dotenv
# 使用する LLM（google-genai | openai | anthropic | ollama | openai-compatible）
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

アバターモード（PIC=PNG表示 / FISH=Boids表示）を選ぶには以下を追加します。

```dotenv
# アバターモード（PIC or FISH）
AVATAR_MODE=PIC
```

- `transport` は `sse` もしくは `stdio` を指定できます。
  - `sse` の場合は `url` が必須
  - `stdio` の場合は `command`（必要なら `args`）を指定
- `settings.py` 側で URL のよくある誤記（`http:/` → `http://` など）を軽微に補正します。

LLM プロバイダ別キー例（必要に応じて設定）:

プロバイダ別設定凡例（要点）
- `google-genai`: 必須 `GEMINI_API_KEY`, モデル `MODEL_NAME`
- `openai`: 必須 `OPENAI_API_KEY`, モデル `OPENAI_MODEL`（任意、既定 `gpt-4o`）, 任意 `OPENAI_BASE_URL`
- `anthropic`: 必須 `ANTHROPIC_API_KEY`, モデル `ANTHROPIC_MODEL`
- `ollama`: 必須 `OLLAMA_BASE_URL`（既定 `http://localhost:11434/v1`）, モデル `OLLAMA_MODEL`（例 `llama3.1:8b`）
- `openai-compatible`（LiteLLM 等）: 必須 `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, モデル `OPENAI_COMPAT_MODEL`

```dotenv
# OpenAI を使う場合
LLM_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1  # 任意

# Anthropic を使う場合
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Ollama を使う場合（OpenAI互換 /v1 経由）
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b
# OLLAMA_API_KEY=ollama   # 未使用でもOK

# OpenAI 互換 API（LiteLLM など）を使う場合
LLM_PROVIDER=openai-compatible
OPENAI_COMPAT_BASE_URL=http://localhost:4000/v1
OPENAI_COMPAT_API_KEY=sk-...
OPENAI_COMPAT_MODEL=gpt-4o
```

UI関連の主な可変値:
- `AVATAR_NAME`, `AVATAR_FULL_NAME`（アバター名）
- `AVATAR_IMAGE_IDLE`, `AVATAR_IMAGE_TALK`（画像が無い場合は ASCII にフォールバック）
- `TYPEWRITER_DELAY_MS`, `MOUTH_ANIMATION_INTERVAL_MS`（表示/口パク速度）
- `AVATAR_MODE`（`PIC`/`FISH`）

## 起動方法
```bash
# ターミナル UI（通常）
python terminal_app_strands.py

# Boids 魚群表示を組み込み済みの UI（右パネルのアバター表示が PIC/FISH に対応）
python terminal_app_fish.py
```
起動時、上部に以下のようなステータスを表示します。
- 例: `Shoebill Communicator オンライン | LLM: google-genai | MCP: ✓ (1 connected / 4 configured) | Avatar: image`
  - configured: .env に設定された MCP サーバ数
  - connected: 実際に接続できたサーバ数（ツールが取得できたもの）
  - Avatar: `image`/`ascii`（PICモード）または `boids`（FISHモード）

## 使い方
- 画面下部の入力欄にメッセージを入力して Enter
- 右側のアバターが発話モードになり、左側ログに Markdown 描画で応答が流れます
- ショートカット: `q` または `Ctrl+C` で終了

スラッシュコマンド:
- `/clear`: 画面をクリアして初期ステータスを再表示
- `/mcp`: 利用可能な MCP ツール一覧を Markdown で表示（サーバ別にグルーピング）

## FISH（Boids）モードについて
FISH モードでは、右側のアバターパネルに Boids（魚群行動）を表示します。水槽の壁は描画しません。画面サイズに合わせて自動的に群れの行動領域がフィットします。

初期パラメータ（組み込みの既定値）:
- count: 100（コード側で設定）
- fps: 15
- align: 12, cohere: 14, separate: 7
- max-speed: 3.2, max-force: 0.10
- restitution: 0.95

チャット状態との連動（アニメーションモード）:
- idle: 待機時。ゆっくり散開、青系の色調
- thinking: 送信直後〜応答開始前。中央に集まり渦巻く、黄系の脈動
- answering: 応答ストリーム中。速度アップ＆整列強化、右方向へ流れる、緑系

魚の数を増やす/減らす（調整方法）:
- `terminal_app_fish.py` の Boids 初期化箇所で `BoidsParams(count=...)` を変更してください。
  - 参照: `terminal_app_fish.py:891` 付近（`BoidsParams` の生成）
- 画面のセル数が上限のため、同じセルに重なった魚は密度文字（*, ▓, █）として集約表示されます。

## LLM 切替
- `.env` の `LLM_PROVIDER` を切り替えます
  - `google-genai`: `GEMINI_API_KEY` と `MODEL_NAME` を利用
  - `openai`: `OPENAI_API_KEY` と `OPENAI_MODEL`（任意、デフォルト `gpt-4o`）。`OPENAI_BASE_URL` も指定可
  - `anthropic`: `ANTHROPIC_API_KEY` と `ANTHROPIC_MODEL`（任意）
  - `ollama`: `OLLAMA_BASE_URL`（デフォルト `http://localhost:11434/v1`）, `OLLAMA_MODEL`
  - `openai-compatible`: `OPENAI_COMPAT_BASE_URL`, `OPENAI_COMPAT_API_KEY`, `OPENAI_COMPAT_MODEL`

注意（MCP と使用モデル）:
- Strands Agent の MCP 統合は OpenAI互換モデル（`strands.models.openai.OpenAIModel`）を用いますが、.env の `LLM_PROVIDER` に合わせて以下のようにモデルを選択します。
  - `openai`: `OPENAI_MODEL` と `OPENAI_API_KEY`（任意で `OPENAI_BASE_URL`）
  - `openai-compatible`: `OPENAI_COMPAT_MODEL` と `OPENAI_COMPAT_API_KEY`（`OPENAI_COMPAT_BASE_URL`）
  - `ollama`: `OLLAMA_MODEL` と `OLLAMA_BASE_URL`（APIキーは未使用でも可）
  - 上記以外（`google-genai` / `anthropic`）の場合、環境に対応 Strands モデルが無く、かつ OpenAI 互換の情報も `.env` に無い場合は MCP を利用できず、接続失敗メッセージとなります。

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

ヒント:
- `.env` の `DEBUG_MODE=True` で `settings.py` の読み込み・パース状況を標準出力に表示（UIは静かなまま）。
- URL のよくある誤記（`http:/` や `https:/`）は自動補正されます。

## トラブルシュート
- MCP が `✓ (1 connected / 4 configured)` のように一部しか接続されない
  - 対象サーバの SSE エンドポイントが生きているか（`curl http://localhost:PORT/sse`）
  - URL パス（`/sse` か `/mcp` か）と認証ヘッダの有無
  - stdio の場合は `command` の実行パスと権限
- 画面が真っ黒／ログが出ない
  - ほとんどの初期化ログは抑制済み。`.env` の `DEBUG_MODE=True` にしてから `settings.py` の標準出力を確認
- Ascii アートが表示されない
  - `static/images/` に `AVATAR_IMAGE_IDLE`, `AVATAR_IMAGE_TALK` が無い場合は ASCII に自動フォールバックします

サウンド（タイプ音）について:
- `numpy` と `pygame` が利用可能な場合に自動有効化されます（利用不可なら自動で無効）。
- 音量や減衰は以下の値で調整できます。

```dotenv
BEEP_FREQUENCY_HZ=800
BEEP_DURATION_MS=50
BEEP_VOLUME=0.05
BEEP_VOLUME_END=0.01
```
- 有効化のためのインストール例: `pip install numpy pygame`

## 主要ファイル
- `terminal_app_strands.py`：ターミナル UI 本体（Strands + MCP + LLM 統合）
- `terminal_app_fish.py`：PIC/FISH（Boids）対応 UI。本ドキュメントの FISH 節参照
- `boids_textual.py`：Boids 表示ロジック（組み込み版）
- `settings.py`：`.env` 読み込みと検証、MCP サーバ設定のパース
- `.env`：環境変数による設定
 - `sound_manager.py`：タイプ音（任意機能、依存があれば自動有効化）

## ライセンス
- 本リポジトリの `LICENSE` を参照してください
