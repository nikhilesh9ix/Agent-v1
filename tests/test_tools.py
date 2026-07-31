"""Tests for the tool registry and schema inference (no network required)."""

import pytest

from agent_framework.tools import ToolRegistry, build_tool_from_function


def sample(a: int, b: int, label: str = "sum") -> str:
    """Add two integers.

    Args:
        a: first number.
        b: second number.
        label: name for the result.
    """
    return f"{label}={a + b}"


def test_schema_inference_types_and_required():
    tool = build_tool_from_function(sample)
    props = tool.parameters["properties"]
    assert props["a"]["type"] == "integer"
    assert props["label"]["type"] == "string"
    assert props["a"]["description"] == "first number."
    # Defaulted params are optional.
    assert tool.parameters["required"] == ["a", "b"]
    assert tool.description == "Add two integers."


def test_dispatch_success():
    reg = ToolRegistry()
    reg.tool(sample)
    assert reg.dispatch("sample", {"a": 21, "b": 21}) == "sum=42"


def test_dispatch_parses_json_string_arguments():
    reg = ToolRegistry()
    reg.tool(sample)
    assert reg.dispatch("sample", '{"a": 1, "b": 2}') == "sum=3"


def test_dispatch_unknown_tool():
    reg = ToolRegistry()
    assert reg.dispatch("nope", {}).startswith("ERROR: unknown tool")


def test_dispatch_bad_arguments():
    reg = ToolRegistry()
    reg.tool(sample)
    assert reg.dispatch("sample", {"a": 1}).startswith("ERROR: bad arguments")


def test_dispatch_null_arguments_treated_as_empty():
    reg = ToolRegistry()

    @reg.tool
    def now() -> str:
        """Return a constant."""
        return "ok"

    assert reg.dispatch("now", "null") == "ok"


def test_duplicate_registration_raises():
    reg = ToolRegistry()
    reg.tool(sample)
    with pytest.raises(ValueError):
        reg.tool(sample)
