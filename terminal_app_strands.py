"""
Textualベースターミナルチャットアプリケーション
mcp-useライブラリを使用したマルチLLMプロバイダ、マルチMCPサーバ対応版
"""
import asyncio
from pathlib import Path
import io
import sys
import traceback
import logging
import os
import re
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# .envファイル読み込み
load_dotenv()

# =============================================================================
# Strands Agentsのインポートとログ設定
# =============================================================================

# Strands Agentsの正しいインポート
from strands.agent import Agent
from strands.tools.mcp import MCPClient
# MCP transports: stdio + SSE（SSEは動的インポートで両方の命名に対応）
from mcp import stdio_client, StdioServerParameters
try:
    # mcp>=1 の一般的な構成
    from mcp import sse_client  # type: ignore
except Exception:
    try:
        # 古い/別パッケージ構成
        from mcp.client.sse import sse_client  # type: ignore
    except Exception:
        sse_client = None  # ランタイムで検出
from strands.models.openai import OpenAIModel

# 既存のログ設定も維持（追加の保険として）
logging.getLogger('mcp').setLevel(logging.CRITICAL + 1)
logging.getLogger('strands').setLevel(logging.CRITICAL + 1)

# 全体的なログ無効化
logging.disable(logging.CRITICAL)  # すべてのログを無効にする



# 標準出力・標準エラー出力を完全に抑制するクラス（test_mcp2.pyの方法を採用）
class LogSuppressor:
    def __init__(self, stdout: bool = True, stderr: bool = True):
        # 抑止対象の選択（Textual動作を妨げないためstderrのみ抑止も可能）
        self.suppress_stdout = stdout
        self.suppress_stderr = stderr
        self.original_stdout = None
        self.original_stderr = None
        self.devnull = None
        
    def __enter__(self):
        # /dev/nullを開く
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        
        # 必要な方だけ退避・差し替え
        if self.suppress_stdout:
            self.original_stdout = os.dup(1)
            os.dup2(self.devnull, 1)
        if self.suppress_stderr:
            self.original_stderr = os.dup(2)
            os.dup2(self.devnull, 2)
        
        # すべてのログを無効化
        logging.getLogger().setLevel(logging.CRITICAL + 1)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # 差し替えた方のみ元に戻す
            if self.original_stdout is not None:
                os.dup2(self.original_stdout, 1)
                os.close(self.original_stdout)
            if self.original_stderr is not None:
                os.dup2(self.original_stderr, 2)
                os.close(self.original_stderr)
            if self.devnull is not None:
                os.close(self.devnull)
        except Exception:
            pass  # クリーンアップエラーは無視

# LangChain imports（Strands互換性のため維持）
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

try:
    from ascii_magic import AsciiArt
    ASCII_MAGIC_AVAILABLE = True
except ImportError:
    ASCII_MAGIC_AVAILABLE = False

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
try:
    # Textual >= 0.36 provides Center container for easy centering
    from textual.containers import Center as CenterContainer  # type: ignore
except Exception:
    CenterContainer = None  # type: ignore
from textual.widgets import Input, Static, RichLog
from textual.reactive import reactive
from textual.message import Message

import settings
from rich.markdown import Markdown
from rich.console import Group
from rich.text import Text
try:
    from rich.image import Image as RichImage  # Rich >= 13.3 + Pillow が必要
    RICH_IMAGE_AVAILABLE = True
except Exception:
    RichImage = None  # type: ignore
    RICH_IMAGE_AVAILABLE = False

# rich-pixels（ピクセルレンダリング）
try:
    from rich_pixels import Pixels  # type: ignore
    RICH_PIXELS_AVAILABLE = True
except Exception:
    Pixels = None  # type: ignore
    RICH_PIXELS_AVAILABLE = False
from rich.align import Align


class LLMProvider:
    """LLMプロバイダの管理クラス"""
    
    def __init__(self):
        self.llm = None
        self.mcp_clients = {}
        # 実際に接続が成功したMCPサーバ名一覧（UI表示用）
        self.mcp_active_servers = []
        self.strands_agent = None
        self._initialize_llm()
        self._initialize_strands_mcp()
    
    def _initialize_llm(self):
        """LLMプロバイダの初期化"""
        try:
            llm_config = settings.get_current_llm_config()
            provider = settings.LLM_PROVIDER
            
            if provider == 'google-genai':
                self.llm = ChatGoogleGenerativeAI(
                    model=llm_config['model'],
                    google_api_key=llm_config['api_key'],
                    temperature=0.7
                )
            elif provider == 'openai':
                self.llm = ChatOpenAI(
                    model=llm_config['model'],
                    api_key=llm_config['api_key'],
                    temperature=0.7
                )
            elif provider == 'anthropic':
                self.llm = ChatAnthropic(
                    model=llm_config['model'],
                    api_key=llm_config['api_key'],
                    temperature=0.7
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
            # デバッグ情報は削除（ログ非表示化）
            
        except Exception as e:
            # ログにのみ記録（UIには出さない）
            logging.getLogger(__name__).debug("Failed to initialize LLM provider: %s", e)
            raise
    
    def _initialize_strands_mcp(self):
        """Strands Agent MCPクライアントの初期化"""
        try:
            if not settings.MCP_ENABLED or not settings.MCP_SERVERS:
                return
            
            # Strands MCP複数クライアントを初期化（接続時の標準出力ログも抑止）
            with LogSuppressor(stdout=True, stderr=True):
                # 複数のMCPクライアントを作成
                self.mcp_clients = {}
                
                if settings.MCP_SERVERS:
                    for server_name, server_config in settings.MCP_SERVERS.items():
                        try:
                            # 交通手段の自動判定: urlがあればSSE、なければstdio
                            transport = server_config.get("transport") or server_config.get("mode")
                            if not transport:
                                transport = "sse" if "url" in server_config else "stdio"

                            if transport.lower() == "sse":
                                if sse_client is None:
                                    raise RuntimeError("mcp.sse_client is not available in this environment")
                                url = server_config.get("url")
                                headers = server_config.get("headers", {}) or {}
                                if not url:
                                    raise ValueError(f"SSE transport for '{server_name}' requires 'url'")

                                # SSEクライアント（非同期コンテキストマネージャ）を返すファクトリ
                                client = MCPClient(lambda u=url, h=headers: sse_client(u, headers=h))
                            else:
                                # STDIO
                                command = server_config.get("command", "sh")
                                args = server_config.get("args", [])
                                client = MCPClient(lambda cmd=command, a=args: stdio_client(
                                    StdioServerParameters(command=cmd, args=a)
                                ))

                            self.mcp_clients[server_name] = client

                        except Exception as e:
                            logging.getLogger(__name__).debug("Failed to initialize MCP server '%s': %s", server_name, e)
                            continue
                
                # OpenAIModel（Strands用）の初期化
                model_openai = OpenAIModel(
                    client_args={
                        "api_key": os.getenv("OPENAI_API_KEY"),
                    },
                    model_id="gpt-4o",
                    params={
                        "max_tokens": 1000,
                        "temperature": 0.7,
                    }
                )
                
                # 公式ドキュメントに従い、複数のMCPクライアントのツールを統合
                all_tools = []
                
                # 全てのMCPクライアントのコンテキスト内で統合
                mcp_context_managers = list(self.mcp_clients.values())

                if mcp_context_managers:
                    # 複数のMCPクライアントを同時に使用
                    from contextlib import ExitStack
                    
                    with ExitStack() as stack:
                        # 個別にコンテキストを開き、失敗したものはスキップ
                        active_clients = {}
                        for server_name, client in self.mcp_clients.items():
                            try:
                                stack.enter_context(client)
                                active_clients[server_name] = client
                            except Exception as e:
                                logging.getLogger(__name__).debug("Failed to enter context for %s: %s", server_name, e)
                                continue

                        # UI表示用に接続成功サーバを保持
                        self.mcp_active_servers = list(active_clients.keys())
                        
                        # 各クライアントからツールを取得して統合
                        for server_name, client in active_clients.items():
                            try:
                                tools = client.list_tools_sync()
                                all_tools.extend(tools)
                                logging.getLogger(__name__).debug("Loaded %d tools from %s", len(tools), server_name)
                            except Exception as e:
                                logging.getLogger(__name__).debug("Failed to load tools from %s: %s", server_name, e)
                    
                    # Strands Agentの作成（統合されたツールを使用）
                    if all_tools:
                        self.strands_agent = Agent(
                            model=model_openai,
                            tools=all_tools
                        )
                        logging.getLogger(__name__).debug("Created agent with %d total tools", len(all_tools))
                    else:
                        raise ValueError("MCPツールの取得に失敗しました")
                else:
                    raise ValueError("MCP_SERVERS設定が見つかりません")
            
        except Exception as e:
            logging.getLogger(__name__).debug("Failed to initialize Strands MCP: %s", e)
            # MCPが失敗してもLLMは使えるようにする
            self.mcp_clients = {}
            self.strands_agent = None
            self.mcp_active_servers = []
    
    async def generate_response(self, message: str) -> str:
        """メッセージに対する応答を生成（Strands Agent対応）"""
        try:
            # Strands Agent対応の場合
            if self.strands_agent and self.mcp_clients:
                # Textual描画を維持するためstderrのみ抑止
                with LogSuppressor(stdout=False, stderr=True):
                    # 複数のMCPクライアントを同時使用（公式ドキュメント方式）
                    from contextlib import ExitStack
                    
                    with ExitStack() as stack:
                        # 全てのMCPクライアントをコンテキストマネージャーで開く（失敗はスキップ）
                        active_clients = 0
                        for name, client in self.mcp_clients.items():
                            try:
                                stack.enter_context(client)
                                active_clients += 1
                            except Exception as e:
                                logging.getLogger(__name__).debug("Skip client %s due to enter error: %s", name, e)
                                continue
                        
                        if active_clients == 0:
                            return "MCPクライアントへの接続に失敗しました"
                        
                        # Strands Agentで同期的な応答を生成
                        response = ""
                        stream_fn = getattr(self.strands_agent, "stream_async", None) or getattr(self.strands_agent, "stream", None)
                        if not stream_fn:
                            raise AttributeError("Strands Agent does not support streaming (no stream_async/stream)")
                        agent_stream = stream_fn(message)

                        # 初期ハンドシェイクのみ完全抑止
                        first_event = True
                        while True:
                            try:
                                if first_event:
                                    with LogSuppressor(stdout=True, stderr=True):
                                        event = await agent_stream.__anext__()
                                else:
                                    event = await agent_stream.__anext__()
                            except StopAsyncIteration:
                                break
                            text = event.get("event", {}).get("contentBlockDelta", {}).get("delta", {}).get("text")
                            if text:
                                response += text
                            first_event = False
                        return response.strip() if response else "応答を生成できませんでした"
            
            # Strands非対応の場合は直接LLMを使用
            elif self.llm:
                # システムプロンプトを含むメッセージ
                messages = [
                    HumanMessage(content=f"{settings.SYSTEM_INSTRUCTION} User: {message}")
                ]
                response = await self.llm.ainvoke(messages)
                return response.content
            
            else:
                return "エラー: LLMプロバイダが初期化されていません"
                
        except Exception as e:
            error_msg = f"応答生成中にエラーが発生しました: {str(e)}"
            return error_msg
    
    async def generate_response_stream(self, message: str):
        """Strands Agentによる真のストリーミング応答を生成"""
        try:
            # Strands Agent対応の場合
            if self.strands_agent and self.mcp_clients:
                # 複数のMCPクライアントを同時使用（公式ドキュメント方式）
                from contextlib import ExitStack
                
                with ExitStack() as stack:
                    # 全てのMCPクライアントをコンテキストマネージャーで開く（失敗はスキップ）
                    active_clients = 0
                    for name, client in self.mcp_clients.items():
                        try:
                            stack.enter_context(client)
                            active_clients += 1
                        except Exception as e:
                            logging.getLogger(__name__).debug("Skip client %s due to enter error: %s", name, e)
                            continue
                    if active_clients == 0:
                        # 何も開けなければ通常LLMで代替
                        async def _fallback_stream():
                            messages = [HumanMessage(content=f"{settings.SYSTEM_INSTRUCTION} User: {message}")]
                            result = await self.llm.ainvoke(messages) if self.llm else None
                            yield (result.content if result else "応答を生成できませんでした")
                        async for chunk in _fallback_stream():
                            yield chunk
                        return
                    
                    # Strands Agentのストリーミングを直接使用
                    stream_fn = getattr(self.strands_agent, "stream_async", None) or getattr(self.strands_agent, "stream", None)
                    if not stream_fn:
                        raise AttributeError("Strands Agent does not support streaming (no stream_async/stream)")

                    agent_stream = stream_fn(message)

                    # 初期ハンドシェイク時のみ stdout/stderr を完全にミュート（UIへの漏れ防止）
                    first_event = True
                    while True:
                        try:
                            if first_event:
                                with LogSuppressor(stdout=True, stderr=True):
                                    event = await agent_stream.__anext__()
                            else:
                                event = await agent_stream.__anext__()
                        except StopAsyncIteration:
                            break

                        text = event.get("event", {}).get("contentBlockDelta", {}).get("delta", {}).get("text")
                        if text:
                            # 以降はUI描画を維持するため抑止解除済み
                            yield text
                        first_event = False
            
            # Strands非対応の場合はフォールバック
            elif self.llm:
                response = await self.generate_response(message)
                
                # 文字単位でストリーミングを模擬
                buffer = ""
                for char in response:
                    buffer += char
                    # 単語区切りまたは句読点でチャンクを送信
                    if char in [' ', '。', '、', ' ', '.', ',', '!', '?']:
                        if buffer.strip():
                            yield buffer
                            buffer = ""
                
                # 残りのバッファがあれば送信
                if buffer.strip():
                    yield buffer
            
            else:
                yield "エラー: LLMプロバイダが初期化されていません"
                
        except Exception as e:
            error_msg = f"ストリーミング中にエラーが発生しました: {str(e)}"
            logging.getLogger(__name__).debug("%s", error_msg)
            yield error_msg
    
    async def cleanup(self):
        """リソースの後始末"""
        try:
            if self.mcp_clients:
                # Textual描画を維持するためstderrのみ抑止
                with LogSuppressor(stdout=False, stderr=True):
                    # 複数のStrands MCPクライアントのクリーンアップ
                    for server_name, client in self.mcp_clients.items():
                        try:
                            # 必要に応じて適切なクリーンアップを追加
                            pass
                        except Exception:
                            pass
        except Exception as e:
            # クリーンアップエラーは非表示化
            pass

class ChatHistory(RichLog):
    """チャット履歴表示エリア（行全体を再描画してストリーミングを行内更新）"""
    
    def __init__(self, **kwargs):
        super().__init__(wrap=True, **kwargs)
        # 1行= dict(sender, message, color)
        self.chat_lines = []
        self._streaming_index = None
    
    def _render_all(self):
        self.clear()
        for item in self.chat_lines:
            sender = item.get("sender", "")
            message = item.get("message", "")
            color = item.get("color", "green")
            is_markdown = bool(item.get("markdown", False))
            if sender == "USER":
                text = Text()
                text.append(f"| {sender}> ", style=f"bold {color}")
                text.append(message, style=f"bold {color}")
                text.append("\n")
                self.write(text)
            elif sender in ["| SYSTEM", "ERROR", "DEBUG", "SYSTEM"]:
                text = Text()
                label = "> SYSTEM: " if sender in ["| SYSTEM", "SYSTEM"] else f"> {sender}: "
                text.append(label, style="bold green")
                text.append(message, style="dim green")
                text.append("\n")
                self.write(text)
            else:
                # AIなどの発話。Markdown対応
                label = Text()
                label.append(f"| {sender}>\n", style=f"bold {color}")
                if is_markdown:
                    renderable = Group(label, Markdown(message))
                    self.write(renderable)
                    self.write(Text("\n"))
                else:
                    text = Text()
                    text.append(f"| {sender}> ", style=f"bold {color}")
                    text.append(message, style=f"bold {color}")
                    text.append("\n")
                    self.write(text)
        self.scroll_end()
    
    def add_message(self, sender: str, message: str, color: str = "green", markdown: bool = False):
        self.chat_lines.append({"sender": sender, "message": message, "color": color, "markdown": markdown})
        self._render_all()
    
    def add_system_message(self, message: str):
        self.chat_lines.append({"sender": "SYSTEM", "message": message, "color": "green"})
        self._render_all()
    
    def start_streaming(self, sender: str, color: str = "green", markdown: bool = False):
        # 新しい行を作成して以降はこの行を書き換える
        self._streaming_index = len(self.chat_lines)
        self.chat_lines.append({"sender": sender, "message": "", "color": color, "markdown": markdown})
        self._render_all()
    
    def update_streaming(self, content: str):
        if self._streaming_index is None:
            return
        self.chat_lines[self._streaming_index]["message"] = content
        self._render_all()
    
    def end_streaming(self):
        self._streaming_index = None


class AvatarArt(Static):
    """アバター表示（PNG優先・フォールバックでASCII）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_state = "idle"
        self.ascii_cache = {}  # キャッシュでパフォーマンス向上
        self._initialized = False  # 初期化フラグ
        self._last_width = None
        self.render_mode = "unknown"  # 'image' | 'ascii_magic' | 'fallback'
        self.last_error: str | None = None
        
    def _image_cols(self) -> int:
        try:
            # パネル幅に応じて調整（最低20、最大60）
            width = self.size.width or 40
            return max(20, min(60, int(width) - 4))
        except Exception:
            return 40

    def _generate_renderable(self, state: str):
        """PNGがあればRichのImageで表示。無ければASCIIにフォールバック"""
        # 画像ファイルパス
        image_path = f"./static/images/{settings.AVATAR_IMAGE_IDLE if state == 'idle' else settings.AVATAR_IMAGE_TALK}"

        # まずPNG表示（rich.image）を試す
        if RICH_IMAGE_AVAILABLE and Path(image_path).exists():
            try:
                cols = self._image_cols()
                self.render_mode = "image"
                self.last_error = None
                return RichImage.from_file(image_path, width=cols)
            except Exception:
                # Pillow 未導入などのケースはASCIIへ
                import traceback
                self.last_error = traceback.format_exc(limit=1)

        # rich-pixels があればそれを試す（より互換性の高いピクセル描画）
        if RICH_PIXELS_AVAILABLE and Path(image_path).exists():
            try:
                # メソッド名の互換対応
                make = getattr(Pixels, "from_image_path", None) or getattr(Pixels, "from_image", None)
                if make is None:
                    raise RuntimeError("rich-pixels API not found")
                # from_image は PIL.Image を受け取る可能性がある
                if make.__name__ == "from_image":
                    try:
                        from PIL import Image as PILImage  # type: ignore
                        img = PILImage.open(image_path)
                        # 横幅をざっくり合わせる（列数に近づける）
                        cols = self._image_cols()
                        w, h = img.size
                        if w > 0:
                            new_w = max(20, min(120, cols * 2))
                            new_h = max(10, int(h * (new_w / w)))
                            img = img.resize((new_w, new_h))
                        renderable = Pixels.from_image(img)
                    except Exception:
                        renderable = make(image_path)  # type: ignore[arg-type]
                else:
                    renderable = make(image_path)  # type: ignore[misc]
                self.render_mode = "image"
                self.last_error = None
                return renderable
            except Exception:
                import traceback
                self.last_error = traceback.format_exc(limit=1)

        # ASCII Magic があればそれを使う
        if ASCII_MAGIC_AVAILABLE:
            try:
                if state in self.ascii_cache:
                    self.render_mode = "ascii_magic"
                    self.last_error = None
                    return Text.from_ansi(self.ascii_cache[state])
                my_art = AsciiArt.from_image(image_path) if Path(image_path).exists() else None
                ascii_output = (
                    my_art.to_ascii(columns=50, monochrome=False) if my_art else self._get_fallback_ascii(state)
                )
                self.ascii_cache[state] = ascii_output
                self.render_mode = "ascii_magic"
                self.last_error = None
                return Text.from_ansi(ascii_output)
            except Exception:
                import traceback
                self.last_error = traceback.format_exc(limit=1)

        # 最終フォールバック（固定ASCII）
        self.render_mode = "fallback"
        return Text.from_ansi(self._get_fallback_ascii(state))
    
    def _get_fallback_ascii(self, state: str) -> str:
        """フォールバック用ASCII アート"""
        if state == "idle":
            return """
    ╭─────────╮
    │  ◉   ◉  │
    │    ─    │
    │         │
    ╰─────────╯
    SPECTRA
    (IDLE)
            """
        else: 
            return """
    ╭─────────╮
    │  ◉   ◉  │
    │    ○    │
    │  \\_____/ │
    ╰─────────╯
    SPECTRA
    (TALKING)
            """
    
    def set_state(self, state: str):
        """状態変更と画像/ASCIIの更新"""
        if not self._initialized or self.current_state != state or self._last_width != self.size.width:
            self.current_state = state
            renderable = self._generate_renderable(state)
            # 水平センタリング（ウィジェット幅を使う）
            self.update(Align.center(renderable))
            self._initialized = True
            self._last_width = self.size.width
            

class AvatarDisplay(Container):
    """アバター表示エリア"""
    
    current_state = reactive("idle")  # "idle" or "talk"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.avatar_widget = None
    
    def compose(self) -> ComposeResult:
        """アバター表示コンポーネント構築"""
        yield Static(f"[b green]{settings.AVATAR_NAME.upper()}[/]", classes="avatar-label")

        # アバター本体（センタリング用のコンテナに内包）
        self.avatar_widget = AvatarArt(id="avatar-art", classes="avatar-art")
        if CenterContainer:
            yield CenterContainer(self.avatar_widget, id="avatar-center")
        else:
            yield Container(self.avatar_widget, classes="avatar-center")
    
    def on_mount(self):
        """マウント時の初期化"""
        if self.avatar_widget:
            # 初期状態でアバターを表示
            self.avatar_widget.set_state("idle")
            
    def set_talking(self, talking: bool):
        """発話状態の変更"""
        self.current_state = "talk" if talking else "idle"
        if self.avatar_widget:
            self.avatar_widget.set_state(self.current_state)


class ChatInput(Input):
    """チャット入力欄"""
    
    def __init__(self, **kwargs):
        super().__init__(placeholder="メッセージを入力してください...", **kwargs)


class TerminalChatApp(App):
    """メインアプリケーション"""
    
    CSS = """
    Screen {
        background: black;
        text-style: none;
    }
    
    .main-container {
        background: black;
        border: solid green;
    }
    
    .content-area {
        height: 1fr;
    }
    
    .chat-panel {
        background: black;
        border-right: solid green;
        width: 70%;
    }
    
    ChatHistory {
        scrollbar-size-vertical: 1;
        text-style: none;
    }
    
    .chat-message {
        text-style: none;
    }
    
    .user-message {
        text-style: bold;
    }
    
    .ai-message {
        text-style: none;
    }
    
    .system-message {
        text-style: italic;
    }
    
    .avatar-panel {
        background: black;
        width: 30%;
        padding: 1;
    }

    /* アバターを水平・垂直センタリングするラッパー */
    .avatar-center {
        width: 100%;
        height: 1fr;
        align: center middle;            /* 子要素を中央寄せ */
        content-align: center middle;    /* 一部のレイアウトでの保険 */
    }
    
    .input-panel {
        background: black;
        border-top: solid green;
        height: 5;
        min-height: 5;
        max-height: 5;
    }
    
    .chat-line {
        color: green;
        margin: 0 1;
    }
    
    .bright_blue {
        color: blue;
    }
    
    .avatar-label {
        text-align: center;
        margin-bottom: 1;
    }
    
    .avatar-art {
        padding: 0;
        margin: 0;
        text-align: center;              /* 水平センタリング */
        width: 100%;                     /* パネル幅に合わせる */
    }
    
    ChatInput {
        background: black;
        color: green;
        border: none;
        margin: 0 1;
        text-style: none;
    }
    
    ChatInput:focus {
        border: solid green;
    }
    

    """
    
    BINDINGS = [
        Binding("q", "quit", "終了"),
        Binding("ctrl+c", "quit", "終了"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_processing = False
        self.llm_provider = None
    
    def compose(self) -> ComposeResult:
        """UIコンポーネント構築"""
        # Verticalレイアウトでメインコンテンツと入力欄を分離
        with Vertical(classes="main-container"):
            # 上部: チャット履歴とアバター表示の横分割
            with Horizontal(classes="content-area"):
                with Container(classes="chat-panel"):
                    yield ChatHistory(id="chat-history")
                with Container(classes="avatar-panel"):
                    yield AvatarDisplay(id="avatar-display")
            
            # 下部: 入力欄（固定高さ）
            with Container(classes="input-panel"):
                yield ChatInput(id="chat-input")
    
    async def on_mount(self):
        """アプリ起動時の初期化"""
        chat_history = self.query_one("#chat-history", ChatHistory)
        
        try:
            # LLMプロバイダの初期化（接続ログ抑止のためstdout/stderrともに抑止）
            with LogSuppressor(stdout=True, stderr=True):
                self.llm_provider = LLMProvider()
            
            # 初期化完了メッセージ
            provider_name = settings.LLM_PROVIDER
            # 接続成功サーバのみをカウントして表示（設定上の数ではなく実際の接続数）
            connected = len(self.llm_provider.mcp_active_servers)
            configured = len(self.llm_provider.mcp_clients)
            if settings.MCP_ENABLED and self.llm_provider.strands_agent and connected > 0:
                mcp_status = f"✓ ({connected} connected / {configured} configured)"
            else:
                mcp_status = "✗"

            # アバター描画モード（image/ascii）も表示して診断しやすく
            avatar_display = self.query_one("#avatar-display", AvatarDisplay)
            render_mode = (
                avatar_display.avatar_widget.render_mode
                if avatar_display and getattr(avatar_display, "avatar_widget", None)
                else "unknown"
            )
            mode_label = {
                "image": "image",
                "ascii_magic": "ascii",
                "fallback": "ascii",
            }.get(render_mode, "unknown")
            
            extra = ""
            if mode_label != "image":
                err = (
                    avatar_display.avatar_widget.last_error
                    if avatar_display and getattr(avatar_display, "avatar_widget", None)
                    else None
                )
                if err:
                    extra = " (image fallback)"
            chat_history.add_system_message(
                f"{settings.AVATAR_FULL_NAME} オンライン | LLM: {provider_name} | MCP: {mcp_status} | Avatar: {mode_label}{extra}"
            )
            
        except Exception as e:
            error_msg = f"初期化エラー: {str(e)}"
            logging.getLogger(__name__).debug("%s", error_msg)
            chat_history.add_system_message(f"エラー: {error_msg}")
        
        # 入力欄にフォーカス
        chat_input = self.query_one("#chat-input", ChatInput)
        chat_input.focus()
    
    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted):
        """ユーザー入力処理"""
        if self.is_processing or not event.value.strip():
            return
        
        user_message = event.value.strip()
        event.input.clear()
        
        # ユーザーメッセージを履歴に即座に追加して表示
        chat_history = self.query_one("#chat-history", ChatHistory)
        chat_history.add_message("USER", user_message, "cyan")
        
        # 画面更新を強制して確実にユーザー入力を表示
        self.refresh()
        # より長い待機時間でUI更新を確実にする
        await asyncio.sleep(0.05)
        
        # UI更新を再度確認
        chat_history.scroll_end()  # スクロールを最下部に移動
        self.refresh()
        
        # AI応答処理を即座に開始（ユーザー入力表示との分離）
        asyncio.create_task(self._process_ai_response(user_message))
    
    async def _process_ai_response(self, user_message: str):
        """AI応答の処理（ストリーミング対応）"""
        if not self.llm_provider:
            return
        
        self.is_processing = True
        avatar_display = self.query_one("#avatar-display", AvatarDisplay)
        chat_history = self.query_one("#chat-history", ChatHistory)
        
        try:
            # アバターを発話状態に
            avatar_display.set_talking(True)
            
            # UI更新を確実にするため少し待機
            await asyncio.sleep(0.02)
            
            # ストリーミング応答の生成と表示（ChatHistory内部の1行を更新）
            response_buffer = ""
            chat_history.start_streaming(settings.AVATAR_NAME, "green", markdown=True)
            async for chunk in self.llm_provider.generate_response_stream(user_message):
                if not chunk:
                    continue
                response_buffer += chunk
                chat_history.update_streaming(response_buffer)
                self.refresh()
                await asyncio.sleep(0)
            
            # ストリーム終了（行はそのまま確定表示）
            chat_history.end_streaming()
            if not response_buffer or not response_buffer.strip():
                chat_history.add_message(f"{settings.AVATAR_NAME}", "応答を生成できませんでした", "red", markdown=False)

            
        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            chat_history.add_message("ERROR", error_msg, "red")
            
        finally:
            # アバターを静止状態に
            avatar_display.set_talking(False)
            self.is_processing = False
    



    async def on_exit(self):
        """アプリケーション終了時のクリーンアップ"""
        if self.llm_provider:
            await self.llm_provider.cleanup()


def main():
    """アプリケーション起動"""
    app = TerminalChatApp()
    app.run()


if __name__ == "__main__":
    main()
