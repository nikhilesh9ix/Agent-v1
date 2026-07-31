"""A handful of ready-made tools, plus a registry that bundles them.

These show the range an agent needs: a pure-local fact (the clock), a
computation (unit conversion), a filesystem read (which can fail), and a call to
an external API (which can also fail). Each has a hand-written JSON Schema so you
can see exactly what the model reads; Phase 6 adds a @tool decorator that
generates these from type hints instead.
"""

from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone

from .tools import ToolRegistry


# --- the actual functions -------------------------------------------------

def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# Length units expressed in metres, so any pair converts through metres.
_LENGTH_TO_M = {
    "mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
}


def convert_length(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a length between units (mm, cm, m, km, in, ft, yd, mi)."""
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu not in _LENGTH_TO_M or tu not in _LENGTH_TO_M:
        raise ValueError(
            f"unknown unit; supported: {sorted(_LENGTH_TO_M)}"
        )
    metres = float(value) * _LENGTH_TO_M[fu]
    result = metres / _LENGTH_TO_M[tu]
    return f"{value} {from_unit} = {result:g} {to_unit}"


def read_text_file(path: str, max_chars: int = 2000) -> str:
    """Read a UTF-8 text file from disk and return up to max_chars of it."""
    # Raises FileNotFoundError on a missing file — the registry turns that into
    # an error message the agent can react to, rather than crashing.
    with open(path, "r", encoding="utf-8") as f:
        data = f.read(max_chars + 1)
    if len(data) > max_chars:
        return data[:max_chars] + "\n...[truncated]"
    return data


def get_exchange_rate(base: str, target: str) -> str:
    """Get the current exchange rate from `base` currency to `target` (ISO codes).

    Uses the free, key-less open.er-api.com endpoint.
    """
    url = f"https://open.er-api.com/v6/latest/{base.upper()}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (trusted host)
        payload = json.loads(resp.read().decode())
    if payload.get("result") != "success":
        raise RuntimeError(payload.get("error-type", "rate lookup failed"))
    rate = payload["rates"].get(target.upper())
    if rate is None:
        raise ValueError(f"unknown target currency {target!r}")
    return f"1 {base.upper()} = {rate} {target.upper()}"


# --- registry assembly ----------------------------------------------------

def default_registry() -> ToolRegistry:
    """Build a registry pre-loaded with the built-in tools and their schemas."""
    reg = ToolRegistry()

    reg.register(
        "get_current_datetime",
        "Get the current date and time in UTC (ISO-8601). Takes no arguments. "
        "Use when the user asks what time or date it is now.",
        {"type": "object", "properties": {}, "required": []},
        get_current_datetime,
    )

    reg.register(
        "convert_length",
        "Convert a length measurement from one unit to another. "
        "Supported units: mm, cm, m, km, in, ft, yd, mi.",
        {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The amount to convert."},
                "from_unit": {"type": "string", "description": "Unit to convert from."},
                "to_unit": {"type": "string", "description": "Unit to convert to."},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
        convert_length,
    )

    reg.register(
        "read_text_file",
        "Read the contents of a UTF-8 text file on the local disk. "
        "Use only when the user references a specific file path.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 2000).",
                },
            },
            "required": ["path"],
        },
        read_text_file,
    )

    reg.register(
        "get_exchange_rate",
        "Get the current foreign-exchange rate between two currencies given "
        "their 3-letter ISO codes (e.g. USD, EUR, INR, JPY).",
        {
            "type": "object",
            "properties": {
                "base": {"type": "string", "description": "Base currency ISO code."},
                "target": {"type": "string", "description": "Target currency ISO code."},
            },
            "required": ["base", "target"],
        },
        get_exchange_rate,
    )

    return reg
