"""
設定管理モジュール - .envファイルから全設定を読み込み
"""
import os
import json
from dotenv import load_dotenv

# .envファイルを読み込み
env_loaded = load_dotenv()
print(f"DEBUG: .env file loaded: {env_loaded}")
if not env_loaded:
    print("WARNING: .env file not found or could not be loaded")

# ===========================================
# 必須設定（環境変数が設定されていない場合はエラー）
# ===========================================

# AI設定
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']  # Gemini APIキー
MODEL_NAME = os.environ['MODEL_NAME']  # 使用するGeminiモデル

# ===========================================
# 任意設定（デフォルト値あり）
# ===========================================

# アバター設定
AVATAR_NAME = os.getenv('AVATAR_NAME', 'Spectra')
AVATAR_FULL_NAME = os.getenv('AVATAR_FULL_NAME', 'Spectra Communicator')
AVATAR_IMAGE_IDLE = os.getenv('AVATAR_IMAGE_IDLE', 'idle.png')
AVATAR_IMAGE_TALK = os.getenv('AVATAR_IMAGE_TALK', 'talk.png')
AVATAR_MODE = os.getenv('AVATAR_MODE', 'PIC').upper()  # PIC または FISH

# AI性格設定（AVATAR_NAMEに依存）
SYSTEM_INSTRUCTION = os.getenv(
    'SYSTEM_INSTRUCTION',
    f'あなたは{AVATAR_NAME}というAIアシスタントです。技術的で直接的なスタイルで簡潔に応答してください。回答は短く要点を押さえたものにしてください。'
)

# サーバー設定
SERVER_PORT = int(os.getenv('SERVER_PORT', '5000'))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True').lower() == 'true'

# UI設定
TYPEWRITER_DELAY_MS = int(os.getenv('TYPEWRITER_DELAY_MS', '50'))
MOUTH_ANIMATION_INTERVAL_MS = int(os.getenv('MOUTH_ANIMATION_INTERVAL_MS', '150'))

# サウンド設定
BEEP_FREQUENCY_HZ = int(os.getenv('BEEP_FREQUENCY_HZ', '800'))
BEEP_DURATION_MS = int(os.getenv('BEEP_DURATION_MS', '50'))
BEEP_VOLUME = float(os.getenv('BEEP_VOLUME', '0.05'))
BEEP_VOLUME_END = float(os.getenv('BEEP_VOLUME_END', '0.01'))

# ===========================================
# MCP（Model Context Protocol）設定
# ===========================================

# MCPサーバー設定をJSONから読み込み
def load_mcp_servers():
    """MCP_SERVERS_JSON環境変数からMCPサーバー設定を読み込み"""
    debug_enabled = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    if debug_enabled:
        print("DEBUG: Loading MCP servers configuration...")
    
    # 環境変数の存在確認
    mcp_servers_json = os.getenv('MCP_SERVERS_JSON')
    if mcp_servers_json is None:
        if debug_enabled:
            print("WARNING: MCP_SERVERS_JSON environment variable not set")
        return {}
    
    if not mcp_servers_json.strip():
        if debug_enabled:
            print("WARNING: MCP_SERVERS_JSON environment variable is empty")
        return {}
        
    if debug_enabled:
        print(f"DEBUG: MCP_SERVERS_JSON length: {len(mcp_servers_json)} characters")
        print(f"DEBUG: MCP_SERVERS_JSON first 200 chars: {mcp_servers_json[:200]}...")
    
    try:
        servers = json.loads(mcp_servers_json)
        if debug_enabled:
            print(f"DEBUG: Successfully parsed JSON with {len(servers)} servers")
            print(f"DEBUG: Server names: {list(servers.keys())}")
        
        # 各サーバー設定の基本検証
        for name, config in servers.items():
            if debug_enabled:
                print(f"DEBUG: Validating server '{name}'...")
            if not isinstance(config, dict):
                if debug_enabled:
                    print(f"ERROR: Server '{name}' config is not a dictionary: {type(config)}")
                continue
            # transport: stdio or sse を許容
            transport = (config.get('transport') or config.get('mode') or
                         ('sse' if 'url' in config else 'stdio' if 'command' in config else None))
            if not transport:
                if debug_enabled:
                    print(f"ERROR: Server '{name}' missing required 'url' (sse) or 'command' (stdio)")
                continue
            if debug_enabled:
                print(f"DEBUG: Server '{name}' transport: {transport}")
                if transport == 'sse':
                    # よくある誤記 'http:/localhost' を自動修正
                    url = config.get('url')
                    if isinstance(url, str):
                        if url.startswith('http:/') and not url.startswith('http://'):
                            fixed = url.replace('http:/', 'http://', 1)
                            if debug_enabled:
                                print(f"WARNING: Fixed malformed URL for '{name}': {url} -> {fixed}")
                            config['url'] = url = fixed
                        if url.startswith('https:/') and not url.startswith('https://'):
                            fixed = url.replace('https:/', 'https://', 1)
                            if debug_enabled:
                                print(f"WARNING: Fixed malformed URL for '{name}': {url} -> {fixed}")
                            config['url'] = url = fixed
                    print(f"DEBUG: Server '{name}' url: {config.get('url')}")
                    if 'headers' in config:
                        print(f"DEBUG: Server '{name}' headers: (provided)")
                else:
                    print(f"DEBUG: Server '{name}' command: {config.get('command')}")
                    print(f"DEBUG: Server '{name}' args: {config.get('args', [])}")
            
        return servers
        
    except json.JSONDecodeError as e:
        if debug_enabled:
            print(f"ERROR: MCP_SERVERS_JSON JSON decode error: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
            print(f"Raw JSON content: {mcp_servers_json}")
        return {}
    except Exception as e:
        if debug_enabled:
            print(f"ERROR: Unexpected error parsing MCP_SERVERS_JSON: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
        return {}

MCP_SERVERS = load_mcp_servers()

# MCP有効/無効フラグ
MCP_ENABLED = len(MCP_SERVERS) > 0 and os.getenv('MCP_ENABLED', 'True').lower() == 'true'

# ===========================================
# LLMプロバイダー設定
# ===========================================

# 使用するLLMプロバイダー（google-genai, openai, anthropic, ollama, openai-compatible）
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'google-genai')

# 各プロバイダー用のAPIキー・接続設定（必要に応じて設定）
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')  # 任意: 公式OpenAIでも指定可
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# OpenAI互換（LiteLLM, 単独サーバ, Ollamaの/v1など）
OPENAI_COMPAT_API_KEY = os.getenv('OPENAI_COMPAT_API_KEY', os.getenv('OPENAI_API_KEY'))
OPENAI_COMPAT_BASE_URL = os.getenv('OPENAI_COMPAT_BASE_URL')
OPENAI_COMPAT_MODEL = os.getenv('OPENAI_COMPAT_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o'))

# Ollama（OpenAI互換エンドポイント /v1 を利用）
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1:8b')
OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY', 'ollama')  # 未使用でもOK

# LLMプロバイダー別モデル設定
LLM_MODELS = {
    'google-genai': {
        'model': MODEL_NAME,  # 既存のGeminiモデル設定を流用
        'api_key': GEMINI_API_KEY
    },
    'openai': {
        'model': os.getenv('OPENAI_MODEL', 'gpt-4o'),
        'api_key': OPENAI_API_KEY,
        'base_url': OPENAI_BASE_URL,
    },
    'anthropic': {
        'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'),
        'api_key': ANTHROPIC_API_KEY
    },
    # Ollama は OpenAI 互換の /v1 エンドポイントを利用
    'ollama': {
        'model': OLLAMA_MODEL,
        'api_key': OLLAMA_API_KEY,
        'base_url': OLLAMA_BASE_URL,
    },
    # LiteLLM などの OpenAI 互換 API
    'openai-compatible': {
        'model': OPENAI_COMPAT_MODEL,
        'api_key': OPENAI_COMPAT_API_KEY,
        'base_url': OPENAI_COMPAT_BASE_URL,
    },
}

# MCPエージェント設定
MCP_AGENT_MAX_STEPS = int(os.getenv('MCP_AGENT_MAX_STEPS', '30'))

def get_current_llm_config():
    """現在選択されているLLMプロバイダーの設定を取得"""
    if LLM_PROVIDER not in LLM_MODELS:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
    
    config = LLM_MODELS[LLM_PROVIDER]
    # 一部プロバイダー（ollama など）はAPIキー不要の場合もあるため、
    # base_url と model の存在を優先確認。api_key は任意扱い。
    if LLM_PROVIDER in ('openai', 'openai-compatible'):
        if not config.get('api_key'):
            raise ValueError(f"API key not configured for provider: {LLM_PROVIDER}")
    
    return config
