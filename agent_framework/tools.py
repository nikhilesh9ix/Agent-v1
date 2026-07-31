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

import json
from dataclasses import dataclass
from typing import Callable


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
