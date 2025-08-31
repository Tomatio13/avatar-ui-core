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
import time as _time
import numpy as np
import math

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
# LangChain 依存は削除（Strands モデルのみ使用）

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
try:
    from PIL import Image as PILImage
    from PIL import ImageEnhance as PILImageEnhance
    PIL_AVAILABLE = True
except Exception:
    PILImage = None  # type: ignore
    PILImageEnhance = None  # type: ignore
    PIL_AVAILABLE = False

# サウンド（タイプ音）
try:
    from sound_manager import SoundManager as PySoundManager  # 自作Python版
    SOUND_AVAILABLE = True
except Exception:
    PySoundManager = None  # type: ignore
    SOUND_AVAILABLE = False

# Boids 表示（FISH モード用）
try:
    from boids_textual import BoidsWidget, BoidsParams  # type: ignore
    BOIDS_AVAILABLE = True
except Exception:
    BoidsWidget = None  # type: ignore
    BoidsParams = None  # type: ignore
    BOIDS_AVAILABLE = False


class LLMProvider:
    """LLMプロバイダの管理クラス"""
    
    def __init__(self):
        self.model = None  # Strands 用モデル
        self.mcp_clients = {}
        # 実際に接続が成功したMCPサーバ名一覧（UI表示用）
        self.mcp_active_servers = []
        self.strands_agent = None
        self._initialize_model()
        self._initialize_strands_mcp()
    
    def _initialize_model(self):
        """Strands モデルの初期化（.env の LLM_PROVIDER に準拠）"""
        try:
            provider_env = settings.LLM_PROVIDER
            # OpenAI 互換（OpenAI / LiteLLM / Ollama）
            if provider_env in ("openai", "openai-compatible", "ollama"):
                client_args: dict[str, object] = {}
                model_id: str
                if provider_env == 'openai':
                    if os.getenv("OPENAI_API_KEY"):
                        client_args["api_key"] = os.getenv("OPENAI_API_KEY")
                    if os.getenv("OPENAI_BASE_URL"):
                        client_args["base_url"] = os.getenv("OPENAI_BASE_URL")
                    model_id = os.getenv("OPENAI_MODEL", "gpt-4o")
                elif provider_env == 'openai-compatible':
                    key = os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")
                    if key:
                        client_args["api_key"] = key
                    if os.getenv("OPENAI_COMPAT_BASE_URL"):
                        client_args["base_url"] = os.getenv("OPENAI_COMPAT_BASE_URL")
                    model_id = os.getenv("OPENAI_COMPAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
                else:  # ollama
                    client_args["api_key"] = os.getenv("OLLAMA_API_KEY", "ollama")
                    client_args["base_url"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                    model_id = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

                self.model = OpenAIModel(
                    client_args=client_args,
                    model_id=model_id,
                    params={"max_tokens": 1000, "temperature": 0.7},
                )
                return

            # Anthropic（Strands AnthropicModel があれば利用）
            if provider_env == 'anthropic':
                try:
                    from strands.models.anthropic import AnthropicModel as _AnthropicModel  # type: ignore
                    if os.getenv("ANTHROPIC_API_KEY"):
                        self.model = _AnthropicModel(
                            client_args={"api_key": os.getenv("ANTHROPIC_API_KEY")},
                            model_id=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                            params={"max_tokens": 1000, "temperature": 0.7},
                        )
                        return
                except Exception:
                    pass

            # Google/Gemini（対応モデルがあれば利用）
            if provider_env == 'google-genai':
                for mod_name, cls_name in [
                    ("strands.models.google", "GoogleModel"),
                    ("strands.models.gemini", "GeminiModel"),
                    ("strands.models.google_ai", "GoogleAIModel"),
                ]:
                    try:
                        mod = __import__(mod_name, fromlist=[cls_name])
                        GoogleModel = getattr(mod, cls_name)
                        if os.getenv("GEMINI_API_KEY"):
                            self.model = GoogleModel(
                                client_args={"api_key": os.getenv("GEMINI_API_KEY")},
                                model_id=os.getenv("MODEL_NAME"),
                                params={"max_tokens": 1000, "temperature": 0.7},
                            )
                            return
                    except Exception:
                        continue

            # 上記で作れない場合は、OpenAI 互換にフォールバック（あれば）
            ca = {}
            key = os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY")
            if key:
                ca["api_key"] = key
            if os.getenv("OPENAI_COMPAT_BASE_URL"):
                ca["base_url"] = os.getenv("OPENAI_COMPAT_BASE_URL")
            elif os.getenv("OPENAI_BASE_URL"):
                ca["base_url"] = os.getenv("OPENAI_BASE_URL")
            model_id = os.getenv("OPENAI_COMPAT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
            if ca:
                self.model = OpenAIModel(client_args=ca, model_id=model_id, params={"max_tokens": 1000, "temperature": 0.7})

        except Exception as e:
            logging.getLogger(__name__).debug("Failed to initialize Strands model: %s", e)
            self.model = None
    
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
                
                # ここではモデルは作らない。_initialize_model で作成済みの self.model を必ず再利用する
                if self.model is None:
                    raise ValueError("Strandsモデルが初期化されていません (.env を確認してください)")
                
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
                    # すでに初期化済みの Strands モデルを利用
                    agent_model = self.model
                    if all_tools and agent_model is not None:
                        self.strands_agent = Agent(model=agent_model, tools=all_tools)
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
        """メッセージに対する応答を生成（Strands Agent前提）"""
        try:
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
            else:
                return "MCPクライアントへの接続に失敗しました"
                
        except Exception as e:
            error_msg = f"応答生成中にエラーが発生しました: {str(e)}"
            return error_msg
    
    async def generate_response_stream(self, message: str):
        """Strands Agentによる真のストリーミング応答を生成（Strands前提）"""
        try:
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
                        yield "MCPクライアントへの接続に失敗しました"
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

    def list_mcp_tools(self):
        """現在接続中のMCPツール一覧を取得（server, name, description）"""
        results = []
        if not self.mcp_clients:
            return results
        from contextlib import ExitStack
        
        def _extract_name_desc(t):
            # dict-like
            if isinstance(t, dict):
                name = t.get("name") or t.get("tool_name") or t.get("id") or "unknown"
                desc = t.get("description") or ""
                return str(name), str(desc)
            # attribute-based (e.g., MCPAgentTool)
            for attr in ("name", "tool_name", "id"):
                if hasattr(t, attr):
                    name_val = getattr(t, attr)
                    if isinstance(name_val, (str, bytes)):
                        name = name_val.decode() if isinstance(name_val, bytes) else name_val
                        # description candidates
                        desc = ""
                        for d_attr in ("description", "desc", "summary"):
                            if hasattr(t, d_attr):
                                try:
                                    d_val = getattr(t, d_attr)
                                    desc = d_val if isinstance(d_val, str) else str(d_val)
                                except Exception:
                                    pass
                                break
                        return name, desc
            # spec container
            if hasattr(t, "spec"):
                try:
                    spec = getattr(t, "spec")
                    name = getattr(spec, "name", None) or getattr(spec, "id", None) or "unknown"
                    desc = getattr(spec, "description", "")
                    return str(name), str(desc)
                except Exception:
                    pass
            # string fallback
            s = str(t)
            return s, ""

        with ExitStack() as stack:
            active = {}
            for name, client in self.mcp_clients.items():
                try:
                    stack.enter_context(client)
                    active[name] = client
                except Exception:
                    continue
            for server, client in active.items():
                try:
                    tools = client.list_tools_sync()
                    for t in tools:
                        tool_name, tool_desc = _extract_name_desc(t)
                        results.append((server, tool_name, tool_desc))
                except Exception:
                    continue
        return results

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


class TextSpectrum(Static):
    """AIテキストの到着タイミングに同期して発光する簡易スペクトラム。

    時間で1文字ずつ進めるのではなく、feed_text() が呼ばれた瞬間に
    パルスを加算し、フレームごとに減衰させます。よってテキスト表示と
    同期し、終了後は素早く減衰して止まります。
    """

    def __init__(self, fps: int = 45, smoothing: float = 0.6, decay_tau: float = 0.18, gain: float = 1.8, **kwargs):
        super().__init__(**kwargs)
        self.fps = max(10, int(fps))
        self.smoothing = float(smoothing)
        self.decay_tau = float(decay_tau)
        self.gain = float(gain)
        self._ema: np.ndarray | None = None
        self._task: asyncio.Task | None = None
        self._jitter_phase = 0.0
        self._finishing = False

        # 文字カテゴリ集合（簡易）
        self._vowels = set(list("あいうえおアイウエオぁぃぅぇぉゃゅょァィゥェォャュョーaeiouAEIOU"))
        self._nasals = set(list("んン"))
        self._pause = set(list("。、．，・…！？!?。,:;；、\n"))
        self._hard = set(list("かきくけこさしすせそたちつてとカキクケコサシスセソタチツテトぱぴぷぺぽバビブベボパピプペポ"))
        self._soft = set(list("なにぬねのまみむめもらりるれろはひふへほやゆよわをがぎぐげござじずぜぞだぢづでどゃゅょゎゐゑ"))

    # ---- API ----
    def feed_text(self, text: str):
        if not text:
            return
        n_bars = self._current_n_bars()
        if n_bars <= 0:
            return
        if self._ema is None or len(self._ema) != n_bars:
            self._ema = np.zeros(n_bars, dtype=float)

        pulse = np.zeros(n_bars, dtype=float)
        for ch in text:
            cls, energy = self._class_energy(ch)
            prof = self._profile_for(cls, n_bars)
            idxs = np.arange(n_bars)
            jitter = 0.12 * np.sin((self._jitter_phase + idxs) * 0.23)
            self._jitter_phase += 0.01
            contrib = np.clip(energy * (0.7*prof + 0.3*(prof + jitter)) * self.gain, 0.0, 1.0)
            pulse = np.maximum(pulse, contrib)

        # 即時に盛り上げ、既存値も残す
        self._ema = np.clip(np.maximum(self._ema * 0.5, pulse), 0.0, 1.0)
        self._finishing = False

    def clear_text(self):
        if self._ema is not None:
            self._ema *= 0.0
        self._finishing = False

    def finish(self):
        self._finishing = True

    # ---- 内部ロジック ----
    def _current_n_bars(self) -> int:
        w = max(10, int(self.size.width or 40))
        return max(10, w - 2)

    def _class_energy(self, ch: str) -> tuple[str, float]:
        if ch in self._pause:
            return "pause", 0.05
        if ch in self._vowels:
            return "vowel", 0.8
        if ch in self._nasals:
            return "nasal", 0.6
        if ch in self._hard:
            return "hard", 0.7
        if ch in self._soft:
            return "soft", 0.65
        if ch in (" ", "\t", "　"):
            return "space", 0.0
        return "soft", 0.5

    def _profile_for(self, cls: str, n_bars: int) -> np.ndarray:
        x = np.linspace(0, 1, n_bars, dtype=np.float32)
        if cls == "vowel":
            prof = np.exp(-((x-0.6)**2)/(2*0.1**2)) + 0.4*np.exp(-((x-0.3)**2)/(2*0.15**2))
        elif cls == "nasal":
            prof = np.exp(-((x-0.25)**2)/(2*0.07**2))
        elif cls == "hard":
            prof = 0.7*np.exp(-((x-0.15)**2)/(2*0.08**2)) + 0.5*np.exp(-((x-0.55)**2)/(2*0.12**2))
        elif cls == "soft":
            prof = 0.6*np.exp(-((x-0.45)**2)/(2*0.12**2))
        else:
            prof = np.zeros_like(x)
        m = float(np.max(prof) + 1e-6)
        return prof / m

    async def on_mount(self):
        self._task = asyncio.create_task(self._run_loop())

    async def on_unmount(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    async def _run_loop(self):
        last = _time.monotonic()
        while True:
            now = _time.monotonic()
            wait = (1.0/self.fps) - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            now = _time.monotonic()
            dt = now - last
            last = now

            n_bars = self._current_n_bars()
            h = max(4, int(self.size.height or 8))
            if self._ema is None or len(self._ema) != n_bars:
                self._ema = np.zeros(n_bars, dtype=float)

            # 減衰 + 微小ノイズ
            tau = 0.1 if self._finishing else self.decay_tau
            decay = math.exp(-dt / max(1e-3, tau))
            self._ema *= decay
            self._ema = np.clip(self._ema + (np.random.randn(n_bars) * 0.01), 0.0, 1.0)

            # 描画
            rows: list[str] = []
            max_h = max(1, h - 1)
            for y in range(max_h):
                level = y / max(1, max_h)
                row_chars = ["^" if v >= level else " " for v in self._ema[:n_bars]]
                rows.append("".join(row_chars))
            render = "\n".join(reversed(rows)) + "\n"
            self.update(Text(render, style="green"))


class AvatarArt(Static):
    """アバター表示（PNG優先・フォールバックでASCII）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_state = "idle"
        self.ascii_cache = {}  # キャッシュでパフォーマンス向上
        self.image_cache = {}  # 前処理済み画像キャッシュ
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

    def _preprocess_image(self, image_path: str):
        """50x50に縮小し軽くシャープをかけた画像を返す（PIL）。失敗時はNone。"""
        if not PIL_AVAILABLE or not Path(image_path).exists():
            return None
        key = (image_path, 50, 50, 'nearest', 1.05)
        if key in self.image_cache:
            return self.image_cache[key]
        try:
            img = PILImage.open(image_path)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA")
            img = img.resize((50, 50), PILImage.NEAREST)
            if PILImageEnhance is not None:
                img = PILImageEnhance.Sharpness(img).enhance(1.05)
            self.image_cache[key] = img
            return img
        except Exception:
            return None

    def _generate_renderable(self, state: str):
        """PNGがあればRich/Pixelsで表示。無ければASCIIにフォールバック"""
        image_path = f"./static/images/{settings.AVATAR_IMAGE_IDLE if state == 'idle' else settings.AVATAR_IMAGE_TALK}"

        # 1) 可能なら rich-pixels で前処理済み画像をそのまま描画（サイズ忠実）
        if RICH_PIXELS_AVAILABLE and Path(image_path).exists():
            try:
                pre = self._preprocess_image(image_path)
                if pre is not None:
                    self.render_mode = "image"
                    self.last_error = None
                    return Pixels.from_image(pre)
            except Exception:
                import traceback
                self.last_error = traceback.format_exc(limit=1)

        # 2) RichImage で描画（可能なら前処理済みPILを使い、追加スケールを避ける）
        if RICH_IMAGE_AVAILABLE and Path(image_path).exists():
            try:
                pre = self._preprocess_image(image_path)
                if pre is not None and hasattr(RichImage, 'from_pil_image'):
                    self.render_mode = "image"
                    self.last_error = None
                    # width を画像幅に合わせる（追加スケールを避ける）
                    return RichImage.from_pil_image(pre, width=pre.size[0])
                else:
                    cols = self._image_cols()
                    self.render_mode = "image"
                    self.last_error = None
                    return RichImage.from_file(image_path, width=cols)
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
        self.boids_widget = None
        self._anim_task = None
        self._talking = False
    
    def compose(self) -> ComposeResult:
        """アバター表示コンポーネント構築"""
        yield Static(f"[b green]{settings.AVATAR_NAME.upper()}[/]", classes="avatar-label")

        # モードに応じて表示を切替
        if settings.AVATAR_MODE == 'FISH' and BOIDS_AVAILABLE:
            # Boids を生成（指定の初期値。tank_w/h は自動フィットにする）
            params = BoidsParams(
                max_speed=3.2,
                max_force=0.10,
                align_radius=12.0,
                cohere_radius=14.0,
                separate_radius=7.0,
                count=100,
                tank_w=None,
                tank_h=None,
                restitution=0.95,
            )
            self.boids_widget = BoidsWidget(params, draw_border=False, id="boids")
            self.boids_widget.fps_target = 15.0
            # FISH では中央寄せせず、そのまま伸縮
            yield Container(self.boids_widget, id="avatar-center")
        else:
            # アバター本体（センタリング用のコンテナに内包）
            self.avatar_widget = AvatarArt(id="avatar-art", classes="avatar-art")
            if CenterContainer:
                yield CenterContainer(self.avatar_widget, id="avatar-center")
            else:
                yield Container(self.avatar_widget, classes="avatar-center")
        # スペクトラムは AvatarDisplay 内には置かない（親側でレイアウト）
    
    def on_mount(self):
        """マウント時の初期化"""
        if self.avatar_widget:
            # 初期状態でアバターを表示
            self.avatar_widget.set_state("idle")
            
    def set_talking(self, talking: bool):
        """発話状態の変更"""
        # 状態が同じなら何もしない
        if talking == self._talking:
            return
        self._talking = talking
        self.current_state = "talk" if talking else "idle"

        # 既存アニメーションを停止
        if self._anim_task:
            try:
                self._anim_task.cancel()
            except Exception:
                pass
            finally:
                self._anim_task = None

        # FISH モードでは口パクは行わない
        if settings.AVATAR_MODE == 'FISH':
            return

        if talking:
            # 口パクアニメーション開始
            async def _run_animation():
                try:
                    state_open = True
                    interval = max(0.03, (settings.MOUTH_ANIMATION_INTERVAL_MS or 150) / 1000)
                    while self._talking and self.avatar_widget:
                        self.avatar_widget.set_state("talk" if state_open else "idle")
                        state_open = not state_open
                        await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    pass
            self._anim_task = asyncio.create_task(_run_animation())
        else:
            # 停止時は必ずidleに戻す
            if self.avatar_widget:
                self.avatar_widget.set_state("idle")


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

    /* テキストスペクトラム（白文字/黒背景） */
    .text-spectrum {
        background: black;
        color: green;
        height: 14;
        min-height: 6;
        max-height: 10;
        margin-top: 12;     /* 画像のすぐ下に寄せる */
        padding-top: 0;
        overflow: hidden;
    }

    /* アバターを水平・垂直センタリングするラッパー（class と id 両方指定） */
    .avatar-center, #avatar-center {
        width: 100%;
        align: center top;         /* 子要素を上寄せ */
        content-align: center top;
        height: auto;              /* 画像サイズ分だけに縮む（PIC時） */
        margin-bottom: 0;
        padding-bottom: 0;
    }

    /* AvatarDisplay 自体も縮むように（PIC時） */
    #avatar-display {
        height: auto;
        content-align: center top;
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
    

    """ + (
        """
        /* FISH モードでは Boids をできるだけ広く表示 */
        #avatar-display { height: 1fr; }
        .avatar-center, #avatar-center { height: 1fr; align: center middle; content-align: center middle; }
        #boids { width: 100%; height: 1fr; }
        """ if settings.AVATAR_MODE == 'FISH' else ""
    )
    
    BINDINGS = [
        Binding("q", "quit", "終了"),
        Binding("ctrl+c", "quit", "終了"),
    ]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_processing = False
        self.llm_provider = None
        self.sound_manager = None
        self._last_beep_ts = 0.0
    
    def compose(self) -> ComposeResult:
        """UIコンポーネント構築"""
        # Verticalレイアウトでメインコンテンツと入力欄を分離
        with Vertical(classes="main-container"):
            # 上部: チャット履歴とアバター表示の横分割
            with Horizontal(classes="content-area"):
                with Container(classes="chat-panel"):
                    yield ChatHistory(id="chat-history")
                with Container(classes="avatar-panel"):
                    # 右側: アバター + テキストスペクトラム（縦積み）
                    yield AvatarDisplay(id="avatar-display")
                    yield TextSpectrum(id="text-spectrum", classes="text-spectrum", gain=2.2, decay_tau=0.16)
            
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
                # サウンド初期化（依存が無ければ自動的に無効）
                if SOUND_AVAILABLE and PySoundManager is not None:
                    try:
                        self.sound_manager = PySoundManager(settings)
                        # 初回安定化のため待機（ミキサーウォームアップ後）
                        _time.sleep(0.05)
                    except Exception:
                        self.sound_manager = None
            
            # 初期化完了メッセージ
            provider_name = settings.LLM_PROVIDER
            # 接続成功サーバのみをカウントして表示（設定上の数ではなく実際の接続数）
            connected = len(self.llm_provider.mcp_active_servers)
            configured = len(self.llm_provider.mcp_clients)
            if settings.MCP_ENABLED and self.llm_provider.strands_agent and connected > 0:
                mcp_status = f"✓ ({connected} connected / {configured} configured)"
            else:
                mcp_status = "✗"

            # アバター描画モード（image/ascii/boids）を表示
            avatar_display = self.query_one("#avatar-display", AvatarDisplay)
            if settings.AVATAR_MODE == 'FISH':
                mode_label = "boids"
                extra = ""
            else:
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
        
        # スラッシュコマンド処理
        if user_message.startswith("/"):
            cmd = user_message.strip().lower()
            if cmd == "/clear":
                # UIをクリアし、初期ステータスを再表示
                chat_history.chat_lines = []
                chat_history._render_all()
                provider_name = settings.LLM_PROVIDER
                connected = len(self.llm_provider.mcp_active_servers) if self.llm_provider else 0
                configured = len(self.llm_provider.mcp_clients) if self.llm_provider else 0
                mcp_status = (
                    f"✓ ({connected} connected / {configured} configured)"
                    if settings.MCP_ENABLED and self.llm_provider and self.llm_provider.strands_agent and connected > 0
                    else "✗"
                )
                chat_history.add_system_message(
                    f"{settings.AVATAR_FULL_NAME} オンライン | LLM: {provider_name} | MCP: {mcp_status}"
                )
                return
            if cmd == "/mcp":
                tools = []
                if self.llm_provider:
                    try:
                        tools = self.llm_provider.list_mcp_tools()
                    except Exception:
                        tools = []
                if tools:
                    # Markdownで見やすく
                    lines = ["利用可能なMCPツール:"]
                    # グループ化（server毎）
                    by_server = {}
                    for server, name, desc in tools:
                        by_server.setdefault(server, []).append((name, desc))
                    for server, names in by_server.items():
                        lines.append(f"- {server}")
                        # 名前でソート
                        for name, desc in sorted(names, key=lambda x: x[0]):
                            if desc and desc.strip():
                                lines.append(f"  - {name}: {desc}")
                            else:
                                lines.append(f"  - {name}")
                    chat_history.add_message("SYSTEM", "\n".join(lines), markdown=True)
                else:
                    chat_history.add_system_message("MCPツールは見つかりませんでした")
                return

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
        spectrum = None
        try:
            spectrum = self.query_one("#text-spectrum", TextSpectrum)
        except Exception:
            spectrum = None
        
        try:
            # ストリーミング応答の生成と表示（ChatHistory内部の1行を更新）
            response_buffer = ""
            chat_history.start_streaming(settings.AVATAR_NAME, "green", markdown=True)
            if spectrum:
                spectrum.clear_text()
            started_talk = False
            async for chunk in self.llm_provider.generate_response_stream(user_message):
                if not chunk:
                    continue
                if not started_talk:
                    # 実際にテキストが届いたタイミングで口パク開始
                    avatar_display.set_talking(True)
                    started_talk = True
                response_buffer += chunk
                chat_history.update_streaming(response_buffer)
                if spectrum:
                    spectrum.feed_text(chunk)
                # タイプ音（AI応答が実際に表示されている間のみ）
                if self.sound_manager and getattr(self.sound_manager, "is_enabled", lambda: False)():
                    now = _time.monotonic()
                    min_interval = max(0.03, (settings.BEEP_DURATION_MS or 50) / 1000.0 * 0.9)
                    if now - self._last_beep_ts >= min_interval:
                        try:
                            self.sound_manager.play_type_sound()
                            self._last_beep_ts = now
                        except Exception:
                            pass
                self.refresh()
                await asyncio.sleep(0)
            
            # ストリーム終了（行はそのまま確定表示）
            chat_history.end_streaming()
            if not response_buffer or not response_buffer.strip():
                chat_history.add_message(f"{settings.AVATAR_NAME}", "応答を生成できませんでした", "red", markdown=False)
            if spectrum:
                spectrum.finish()

            
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
