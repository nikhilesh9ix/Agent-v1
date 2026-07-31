"""Built-in tools and a ready-made registry: a local fact (clock), a computation
(unit conversion), a filesystem read, and an external API call."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone

from .tools import ToolRegistry


def get_current_datetime() -> str:
    """Return the current UTC date and time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


_LENGTH_TO_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0,
                "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}


def convert_length(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a length between units (mm, cm, m, km, in, ft, yd, mi)."""
    fu, tu = from_unit.lower(), to_unit.lower()
    if fu not in _LENGTH_TO_M or tu not in _LENGTH_TO_M:
        raise ValueError(f"unknown unit; supported: {sorted(_LENGTH_TO_M)}")
    result = float(value) * _LENGTH_TO_M[fu] / _LENGTH_TO_M[tu]
    return f"{value} {from_unit} = {result:g} {to_unit}"


def read_text_file(path: str, max_chars: int = 2000) -> str:
    """Read a UTF-8 text file and return up to max_chars of it."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read(max_chars + 1)
    return data[:max_chars] + "\n...[truncated]" if len(data) > max_chars else data


def get_exchange_rate(base: str, target: str) -> str:
    """Get the current exchange rate from `base` to `target` (ISO codes)."""
    url = f"https://open.er-api.com/v6/latest/{base.upper()}"
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310
        payload = json.loads(resp.read().decode())
    if payload.get("result") != "success":
        raise RuntimeError(payload.get("error-type", "rate lookup failed"))
    rate = payload["rates"].get(target.upper())
    if rate is None:
        raise ValueError(f"unknown target currency {target!r}")
    return f"1 {base.upper()} = {rate} {target.upper()}"


def default_registry() -> ToolRegistry:
    """A registry pre-loaded with the built-in tools (schemas from type hints)."""
    reg = ToolRegistry()
    for fn in (get_current_datetime, convert_length, read_text_file, get_exchange_rate):
        reg.tool(fn)
    return reg
