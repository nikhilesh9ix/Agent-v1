"""Phase 3: the tool-calling machinery.

An LLM can't run code. Given a list of tool *descriptions* (JSON Schema), it can
only reply "I'd like to call function X with arguments Y". Our code is the worker
that actually runs X, then feeds the result back so the model can keep reasoning.

The ToolRegistry is the switchboard: it holds the callable functions plus the
schemas we advertise to the model, and it dispatches a requested call to the
right function — validating and catching failures so a bad call becomes a
message the agent can recover from, not a crash.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from typing import Callable, get_type_hints


@dataclass
class Tool:
    """A callable paired with the JSON-Schema the LLM sees.

    `parameters` is a JSON-Schema object describing the arguments — this exact
    text is what the model reads to decide when and how to call the tool, so its
    descriptions matter as much as the code.
    """

    name: str
    description: str
    parameters: dict
    func: Callable

    def openai_schema(self) -> dict:
        """Shape the tool the way the Chat Completions `tools` param expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registers, advertises, and dispatches tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, parameters: dict,
                 func: Callable) -> None:
        """Add a tool. Name must be unique within the registry."""
        if name in self._tools:
            raise ValueError(f"Tool {name!r} already registered")
        self._tools[name] = Tool(name, description, parameters, func)

    def add(self, tool: Tool) -> None:
        """Register an already-built Tool (used by the @tool decorator)."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def tool(self, func: Callable) -> Callable:
        """Decorator: register `func` as a tool, schema inferred from its hints.

            registry = ToolRegistry()

            @registry.tool
            def add(a: int, b: int) -> str:
                '''Add two integers.'''
                return str(a + b)

        The function is returned unchanged, so it stays directly callable.
        """
        self.add(build_tool_from_function(func))
        return func

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_schemas(self) -> list[dict]:
        """The `tools=` payload for an API request. Empty list -> no tools."""
        return [t.openai_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: str | dict) -> str:
        """Run a tool call and return a string result for the model.

        Failures are *returned*, not raised: an unknown tool, malformed JSON
        arguments, or an exception inside the tool all become an error string.
        That keeps the agent alive — it can read the error and try again — which
        is the whole point of not trusting the model's output blindly.
        """
        if name not in self._tools:
            return f"ERROR: unknown tool {name!r}. Available: {self.names()}"

        # The API hands us arguments as a JSON *string*; accept a dict too.
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return f"ERROR: could not parse arguments for {name!r}: {e}"

        # Some models emit null/None for a no-argument tool — treat as empty.
        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            return f"ERROR: arguments for {name!r} must be a JSON object"

        try:
            result = self._tools[name].func(**arguments)
        except TypeError as e:
            # Wrong/missing/extra kwargs — the model guessed the signature wrong.
            return f"ERROR: bad arguments for {name!r}: {e}"
        except Exception as e:  # noqa: BLE001 — surface any tool failure to the model
            return f"ERROR: tool {name!r} failed: {type(e).__name__}: {e}"

        return result if isinstance(result, str) else json.dumps(result)


# --- schema inference from type hints (Phase 6) ---------------------------

_PY_TO_JSON = {
    str: "string", int: "integer", float: "number",
    bool: "boolean", list: "array", dict: "object",
}


def _parse_arg_docs(docstring: str) -> dict[str, str]:
    """Pull per-arg descriptions from a Google-style `Args:` block, if present."""
    if not docstring or "Args:" not in docstring:
        return {}
    args_block = docstring.split("Args:", 1)[1]
    # Stop at the next section header (Returns:, Raises:, ...).
    args_block = re.split(r"\n\s*\w+:\s*\n", "\n" + args_block, maxsplit=1)[0]
    docs: dict[str, str] = {}
    for match in re.finditer(r"^\s*(\w+)\s*(?:\([^)]*\))?:\s*(.+)$", args_block, re.MULTILINE):
        docs[match.group(1)] = match.group(2).strip()
    return docs


def build_tool_from_function(func: Callable, name: str | None = None,
                             description: str | None = None) -> Tool:
    """Construct a Tool by inspecting a function's signature, hints, and docstring.

    This is what removes the hand-written JSON Schema: the parameter types come
    from annotations, `required` from which params lack defaults, and the
    descriptions from the docstring's summary line and Args block.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    summary = doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    arg_docs = _parse_arg_docs(doc)

    properties: dict[str, dict] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        json_type = _PY_TO_JSON.get(hints.get(pname, str), "string")
        prop = {"type": json_type}
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
    """Standalone decorator: attach an inferred Tool to a function as `.tool_def`.

    Use when you want to build the Tool now but register it into a chosen
    ToolRegistry later (`registry.add(func.tool_def)`). To register in one step,
    prefer the bound `ToolRegistry.tool` decorator instead.
    """
    func.tool_def = build_tool_from_function(func)  # type: ignore[attr-defined]
    return func
