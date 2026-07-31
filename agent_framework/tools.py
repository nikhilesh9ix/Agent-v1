"""Tools: describe functions to the LLM, dispatch the calls it requests.

The model can't run code — it replies "call X with args Y". ToolRegistry holds
the functions plus their JSON schemas and runs the requested call, turning any
failure into an error string the agent can recover from.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Callable, get_type_hints


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema object the LLM reads
    func: Callable

    def openai_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


class ToolRegistry:
    """Registers, advertises, and dispatches tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict, func: Callable) -> None:
        self.add(Tool(name, description, parameters, func))

    def add(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def tool(self, func: Callable) -> Callable:
        """Decorator: register func as a tool, schema inferred from its hints."""
        self.add(build_tool_from_function(func))
        return func

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict]:
        return [t.openai_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: str | dict) -> str:
        """Run a tool call, returning a string result. Failures come back as
        error strings so the agent can react instead of crashing."""
        if name not in self._tools:
            return f"ERROR: unknown tool {name!r}. Available: {self.names()}"

        if isinstance(arguments, str):  # API sends args as a JSON string
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return f"ERROR: could not parse arguments for {name!r}: {e}"
        if arguments is None:  # some models emit null for a no-arg tool
            arguments = {}
        if not isinstance(arguments, dict):
            return f"ERROR: arguments for {name!r} must be a JSON object"

        try:
            result = self._tools[name].func(**arguments)
        except TypeError as e:
            return f"ERROR: bad arguments for {name!r}: {e}"
        except Exception as e:  # noqa: BLE001 — surface any tool failure to the model
            return f"ERROR: tool {name!r} failed: {type(e).__name__}: {e}"
        return result if isinstance(result, str) else json.dumps(result)


# --- schema inference from type hints -------------------------------------

_PY_TO_JSON = {str: "string", int: "integer", float: "number",
               bool: "boolean", list: "array", dict: "object"}


def _parse_arg_docs(docstring: str) -> dict[str, str]:
    """Pull per-arg descriptions from a Google-style `Args:` block."""
    if not docstring or "Args:" not in docstring:
        return {}
    block = docstring.split("Args:", 1)[1]
    block = re.split(r"\n\s*\w+:\s*\n", "\n" + block, maxsplit=1)[0]
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", block, re.MULTILINE)}


def build_tool_from_function(func: Callable, name: str | None = None,
                             description: str | None = None) -> Tool:
    """Build a Tool from a function's signature, type hints, and docstring."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    summary = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    arg_docs = _parse_arg_docs(doc)

    properties, required = {}, []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        prop = {"type": _PY_TO_JSON.get(hints.get(pname, str), "string")}
        if pname in arg_docs:
            prop["description"] = arg_docs[pname]
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return Tool(
        name=name or func.__name__,
        description=description or summary or func.__name__,
        parameters={"type": "object", "properties": properties, "required": required},
        func=func,
    )


def tool(func: Callable) -> Callable:
    """Attach an inferred Tool to a function as `.tool_def` for later registration."""
    func.tool_def = build_tool_from_function(func)  # type: ignore[attr-defined]
    return func
