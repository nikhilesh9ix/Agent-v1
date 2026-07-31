"""Tests for episodic memory and message serialization (no network required)."""

from types import SimpleNamespace

from agent_framework.llm import assistant_message_dict
from agent_framework.memory import SQLiteMemory


def test_sqlite_memory_recency_and_limit(tmp_path):
    mem = SQLiteMemory(str(tmp_path / "mem.db"))
    for i in range(5):
        mem.add(f"summary {i}", {"user_query": f"q{i}"})

    recent = mem.retrieve("anything", k=3)
    assert len(recent) == 3
    # Returned oldest-first, so the newest three are 2, 3, 4.
    assert "summary 2" in recent[0]
    assert "summary 4" in recent[-1]
    mem.close()


def test_sqlite_memory_persists_across_instances(tmp_path):
    path = str(tmp_path / "mem.db")
    m1 = SQLiteMemory(path)
    m1.add("remembered fact", {"user_query": "q"})
    m1.close()

    m2 = SQLiteMemory(path)
    assert any("remembered fact" in s for s in m2.retrieve("q", k=5))
    m2.close()


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
