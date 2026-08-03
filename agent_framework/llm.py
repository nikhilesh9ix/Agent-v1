"""Shared LLM client. All provider/model/auth handling lives here.

Groq and OpenAI both speak the OpenAI Chat Completions protocol, so one client
drives either — selected with the LLM_PROVIDER env var.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_PROVIDERS = {
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
             "default_model": "llama-3.3-70b-versatile"},
}

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"Unknown LLM_PROVIDER={PROVIDER!r}. Options: {list(_PROVIDERS)}")

_CFG = _PROVIDERS[PROVIDER]
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or _CFG["default_model"]


def configure(provider: str | None = None, model: str | None = None) -> None:
    """Switch provider and/or default model at runtime (used by the UI)."""
    global PROVIDER, _CFG, DEFAULT_MODEL
    if provider:
        provider = provider.lower()
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Options: {list(_PROVIDERS)}")
        PROVIDER, _CFG = provider, _PROVIDERS[provider]
        DEFAULT_MODEL = model or _CFG["default_model"]
        get_client.cache_clear()
    elif model:
        DEFAULT_MODEL = model


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Singleton chat client for the active provider."""
    key = os.getenv(_CFG["key_env"])
    if not key:
        raise RuntimeError(f"{_CFG['key_env']} is not set. Add it to .env.")
    return OpenAI(api_key=key, base_url=_CFG["base_url"])


def chat(messages: list[dict], model: str | None = None, **kwargs):
    """Chat Completions call. kwargs (tools, tool_choice, ...) pass straight through."""
    return get_client().chat.completions.create(
        model=model or DEFAULT_MODEL, messages=messages, **kwargs,
    )


def assistant_message_dict(message) -> dict:
    """Serialize an assistant message for re-sending, keeping only the fields
    every provider accepts (Groq rejects the SDK's extra fields like 'annotations')."""
    data: dict = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        data["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls
        ]
    return data
