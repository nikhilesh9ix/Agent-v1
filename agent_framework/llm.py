"""Thin, shared wrapper around an OpenAI-compatible client.

Every other module talks to the LLM through here, so the SDK, model defaults,
auth, and *which provider* live in exactly one place.

Providers
---------
Groq and OpenAI both speak the OpenAI Chat Completions protocol, so a single
`openai` SDK client — just pointed at a different base_url + key — drives either.
Pick one with the LLM_PROVIDER env var ("openai" default, or "groq").

    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_...

Embeddings: Groq does not offer an embeddings endpoint, so embed() always uses
OpenAI (it needs OPENAI_API_KEY). Episodic memory works with any provider;
semantic (vector) memory additionally needs an OpenAI key.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

# Load .env once at import time so scripts don't each have to.
load_dotenv()

# Per-provider defaults. base_url=None means the SDK's built-in (OpenAI) URL.
_PROVIDERS = {
    "openai": {
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
}

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if PROVIDER not in _PROVIDERS:
    raise RuntimeError(f"Unknown LLM_PROVIDER={PROVIDER!r}. Options: {list(_PROVIDERS)}")

_CFG = _PROVIDERS[PROVIDER]

# LLM_MODEL overrides the provider default; keep OPENAI_MODEL as a legacy alias.
DEFAULT_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or _CFG["default_model"]
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def configure(provider: str | None = None, model: str | None = None) -> None:
    """Switch provider and/or default model at runtime (used by the UI).

    Updates module state and clears the cached client so the next call talks to
    the newly selected provider. The matching API key must be present in the env.
    """
    global PROVIDER, _CFG, DEFAULT_MODEL
    if provider:
        provider = provider.lower()
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider {provider!r}. Options: {list(_PROVIDERS)}")
        PROVIDER = provider
        _CFG = _PROVIDERS[provider]
        DEFAULT_MODEL = model or _CFG["default_model"]
        get_client.cache_clear()
    elif model:
        DEFAULT_MODEL = model


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a process-wide singleton chat client for the active provider."""
    key = os.getenv(_CFG["key_env"])
    if not key:
        raise RuntimeError(
            f"{_CFG['key_env']} is not set for provider {PROVIDER!r}. "
            "Copy .env.example to .env and add your key."
        )
    return OpenAI(api_key=key, base_url=_CFG["base_url"])


@lru_cache(maxsize=1)
def get_embed_client() -> OpenAI:
    """Embeddings client — always OpenAI, since Groq has no embeddings endpoint."""
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "Embeddings require OPENAI_API_KEY (Groq has no embeddings API). "
            "Set it in .env, or use episodic (SQLite) memory only."
        )
    return OpenAI(api_key=key)


def chat(messages: list[dict], model: str | None = None, **kwargs):
    """Send a messages array to the Chat Completions API and return the raw response.

    kwargs pass straight through (tools, tool_choice, temperature, max_tokens...),
    so callers get the full API surface without this wrapper knowing about it.
    """
    client = get_client()
    return client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        **kwargs,
    )


def assistant_message_dict(message) -> dict:
    """Serialize an assistant message for re-sending to the API.

    `message.model_dump()` includes newer SDK fields (annotations, refusal,
    audio, function_call) that stricter OpenAI-compatible backends like Groq
    reject with a 400. We keep only the universally accepted fields: role,
    content, and normalized tool_calls.
    """
    data: dict = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return data


def embed(text: str, model: str | None = None) -> list[float]:
    """Turn a piece of text into an embedding vector (via OpenAI)."""
    client = get_embed_client()
    resp = client.embeddings.create(
        model=model or DEFAULT_EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding
