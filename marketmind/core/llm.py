"""
MarketMind AI - LLM Router (Wave 0 F1)

Provider-agnostic chat interface. One ``chat()`` call routes to whichever
backend is configured: Claude API, Claude CLI, DeepSeek, Groq, OpenAI, Ollama.

Backends can be swapped per *role* (research / debate / classify) so that
high-volume agents run on cheap models while critical reasoning stays on
the smart ones.

Config layering (highest precedence first):
  1. ``LLM_BACKEND`` env var (e.g. "claude_api", "deepseek")
  2. ``local.json -> llm.backend``
  3. Default: "claude_api" if any Claude key is present, else "ollama"

Per-role model overrides via ``local.json -> llm.models.<role>``.

Usage::

    from marketmind.core.llm import get_router
    router = get_router()
    text = router.chat(
        [{"role": "user", "content": "Hello"}],
        role="research",
        max_tokens=800,
    )
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parents[2] / "local.json"

# Sensible defaults per role / backend.  Overridable via local.json.
DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "claude_api": {
        "research":  "claude-opus-4-7",
        "debate":    "claude-sonnet-4-6",
        "classify":  "claude-haiku-4-5-20251001",
        "default":   "claude-sonnet-4-6",
    },
    "claude_cli": {
        # claude CLI ignores model arg by design; kept for parity
        "default":   "claude-sonnet-4-6",
    },
    "deepseek": {
        "research":  "deepseek-chat",
        "debate":    "deepseek-chat",
        "classify":  "deepseek-chat",
        "default":   "deepseek-chat",
    },
    "groq": {
        "research":  "llama-3.3-70b-versatile",
        "debate":    "llama-3.3-70b-versatile",
        "classify":  "llama-3.1-8b-instant",
        "default":   "llama-3.3-70b-versatile",
    },
    "openai": {
        "research":  "gpt-4o",
        "debate":    "gpt-4o",
        "classify":  "gpt-4o-mini",
        "default":   "gpt-4o",
    },
    "ollama": {
        "research":  "llama3.1:70b",
        "debate":    "llama3.1:8b",
        "classify":  "llama3.1:8b",
        "default":   "llama3.1:8b",
    },
}

# Where each backend's API key lives in env vars
ENV_KEY: Dict[str, str] = {
    "claude_api": "ANTHROPIC_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "groq":       "GROQ_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "ollama":     "",  # no key
    "claude_cli": "",  # uses CLI's own auth
}

# Where each backend's API key lives in local.json (top-level key name)
# claude_api → "anthropic" preserves backwards compat with the existing config shape
CONFIG_KEY: Dict[str, str] = {
    "claude_api": "anthropic",
    "deepseek":   "deepseek",
    "groq":       "groq",
    "openai":     "openai",
    "ollama":     "ollama",
    "claude_cli": "anthropic",
}

# Default base URLs for OpenAI-compatible providers
BASE_URL: Dict[str, str] = {
    "deepseek": "https://api.deepseek.com/v1",
    "groq":     "https://api.groq.com/openai/v1",
    "openai":   "https://api.openai.com/v1",
    "ollama":   os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
}


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception as e:
        logger.warning(f"local.json load failed in llm.py: {e}")
        return {}


@dataclass
class LLMConfig:
    backend: str = "claude_api"
    models: Dict[str, str] = field(default_factory=dict)
    api_keys: Dict[str, str] = field(default_factory=dict)
    base_urls: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "LLMConfig":
        cfg = _load_config().get("llm", {}) or {}

        # Pick backend: env var > config > auto-detect
        backend = (
            os.environ.get("LLM_BACKEND")
            or cfg.get("backend")
            or _autodetect_backend(cfg)
        )

        if backend not in DEFAULT_MODELS:
            logger.warning(f"Unknown LLM_BACKEND={backend!r}; falling back to claude_api")
            backend = "claude_api"

        # Models: merge defaults under user overrides
        models = dict(DEFAULT_MODELS.get(backend, {}))
        models.update(cfg.get("models", {}) or {})

        # API keys: env > config > none
        keys = {}
        raw_cfg = _load_config()
        for be, env in ENV_KEY.items():
            if env:
                cfg_section = raw_cfg.get(CONFIG_KEY.get(be, be), {}) or {}
                keys[be] = os.environ.get(env, "") or cfg_section.get("api_key", "")

        # base_urls: allow per-backend override in config
        urls = dict(BASE_URL)
        urls.update(cfg.get("base_urls", {}) or {})

        return cls(backend=backend, models=models, api_keys=keys, base_urls=urls)


def _autodetect_backend(cfg: Dict[str, Any]) -> str:
    """When no backend is configured, pick the first one that has a usable key."""
    if os.environ.get("ANTHROPIC_API_KEY") or _load_config().get("anthropic", {}).get("api_key"):
        return "claude_api"
    if shutil.which("claude"):
        return "claude_cli"
    for be in ("deepseek", "groq", "openai"):
        if os.environ.get(ENV_KEY[be]):
            return be
    return "ollama"


# ─────────────────────────────────────────────────────────────────────────────
# Adapters
# ─────────────────────────────────────────────────────────────────────────────

class _Adapter:
    """Base interface every adapter satisfies."""

    name: str = "base"

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        json_mode: bool = False,
        timeout: float = 60.0,
    ) -> str:
        raise NotImplementedError


class ClaudeAPIAdapter(_Adapter):
    name = "claude_api"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat(self, messages, *, model, max_tokens=1024, temperature=0.7,
             system=None, json_mode=False, timeout=60.0):
        client = self._client_lazy()
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return resp.content[0].text


class ClaudeCLIAdapter(_Adapter):
    """Pipes the prompt to the local ``claude`` CLI. Zero API cost for dev."""

    name = "claude_cli"

    def __init__(self):
        self.cli_path = shutil.which("claude")
        if not self.cli_path:
            raise RuntimeError("`claude` CLI not found on PATH; install with `npm install -g @anthropic-ai/claude-code`")

    def chat(self, messages, *, model, max_tokens=1024, temperature=0.7,
             system=None, json_mode=False, timeout=120.0):
        # Flatten conversation into a single prompt for the CLI
        parts = []
        if system:
            parts.append(f"[System]\n{system}\n")
        for m in messages:
            role = m.get("role", "user").capitalize()
            parts.append(f"[{role}]\n{m.get('content', '')}\n")
        prompt = "\n".join(parts).strip()

        try:
            result = subprocess.run(
                [self.cli_path, "-p", prompt],
                capture_output=True, text=True, timeout=timeout,
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI failed (rc={result.returncode}): {result.stderr.strip()}")
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude CLI timed out after {timeout}s")


class OpenAICompatAdapter(_Adapter):
    """Generic adapter for OpenAI-shaped APIs: DeepSeek, Groq, OpenAI, Ollama."""

    def __init__(self, name: str, base_url: str, api_key: str = ""):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def chat(self, messages, *, model, max_tokens=1024, temperature=0.7,
             system=None, json_mode=False, timeout=60.0):
        body_messages = list(messages)
        if system:
            body_messages = [{"role": "system", "content": system}] + body_messages

        payload: Dict[str, Any] = {
            "model": model,
            "messages": body_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        # OpenAI-style response shape
        return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────

class LLMRouter:
    """Single entry point: ``router.chat(messages, role=...)``."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._adapter: Optional[_Adapter] = None

    @property
    def backend(self) -> str:
        return self.cfg.backend

    def _resolve_adapter(self) -> _Adapter:
        if self._adapter is not None:
            return self._adapter
        be = self.cfg.backend
        if be == "claude_api":
            key = self.cfg.api_keys.get("claude_api", "")
            if not key:
                raise RuntimeError("claude_api backend selected but ANTHROPIC_API_KEY/local.json:anthropic.api_key missing")
            self._adapter = ClaudeAPIAdapter(key)
        elif be == "claude_cli":
            self._adapter = ClaudeCLIAdapter()
        elif be in ("deepseek", "groq", "openai", "ollama"):
            self._adapter = OpenAICompatAdapter(
                name=be,
                base_url=self.cfg.base_urls.get(be, BASE_URL[be]),
                api_key=self.cfg.api_keys.get(be, ""),
            )
        else:
            raise RuntimeError(f"Unknown backend: {be}")
        return self._adapter

    def model_for(self, role: str = "default") -> str:
        return self.cfg.models.get(role) or self.cfg.models.get("default") or "claude-sonnet-4-6"

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        role: str = "default",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system: Optional[str] = None,
        json_mode: bool = False,
        timeout: float = 60.0,
    ) -> str:
        adapter = self._resolve_adapter()
        chosen_model = model or self.model_for(role)
        try:
            return adapter.chat(
                messages,
                model=chosen_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                json_mode=json_mode,
                timeout=timeout,
            )
        except Exception as e:
            logger.error(f"LLMRouter.chat failed [{adapter.name}/{chosen_model}]: {e}")
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_router: Optional[LLMRouter] = None

def get_router() -> LLMRouter:
    global _router
    if _router is None:
        _router = LLMRouter(LLMConfig.load())
        logger.info(f"LLM router initialised: backend={_router.cfg.backend}, models={_router.cfg.models}")
    return _router

def reset_router() -> None:
    """Force a fresh load (useful after config changes)."""
    global _router
    _router = None
