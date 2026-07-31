"""Thin, shared wrapper around the OpenAI client.

Every other module talks to the LLM through here, so the SDK, model defaults,
and auth handling live in exactly one place. Swapping providers later means
editing this file only.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

# Load .env once at import time so scripts don't each have to.
load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a process-wide singleton OpenAI client.

    Cached so we reuse one HTTP connection pool instead of building a new
    client per call. Raises a clear error if the API key is missing.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return OpenAI()


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


def embed(text: str, model: str | None = None) -> list[float]:
    """Turn a piece of text into an embedding vector."""
    client = get_client()
    resp = client.embeddings.create(
        model=model or DEFAULT_EMBED_MODEL,
        input=text,
    )
    return resp.data[0].embedding
