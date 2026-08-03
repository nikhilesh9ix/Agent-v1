"""Tests for provider-safe message serialization (no network required)."""

from types import SimpleNamespace

from agent_framework.llm import assistant_message_dict


def test_assistant_message_dict_strips_extra_fields():
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="do_thing", arguments='{"x": 1}'),
    )
    message = SimpleNamespace(
        content=None,
        tool_calls=[tool_call],
        annotations=["should be dropped"],
        refusal=None,
    )
    out = assistant_message_dict(message)
    assert set(out) == {"role", "content", "tool_calls"}
    assert out["tool_calls"][0]["function"]["name"] == "do_thing"


def test_assistant_message_dict_plain_text():
    message = SimpleNamespace(content="hello", tool_calls=None)
    assert assistant_message_dict(message) == {"role": "assistant", "content": "hello"}
