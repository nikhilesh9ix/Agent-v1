# Agent-v1 — an agent framework from scratch

An LLM **agent framework built from scratch** — the planning loop, the
tool-calling protocol, and a layered memory system — with **no** LangChain,
CrewAI, LlamaIndex, or any agent library. The point is to see exactly what those
frameworks do under the hood: an agent is a `while`-loop around an LLM, wired to
tools and memory. That's it. This repo builds that, piece by piece.

## What's inside

```
agent_framework/
├── llm.py            # one shared OpenAI client (chat + embeddings)
├── conversation.py   # ConversationManager: stateful multi-turn chat
├── tools.py          # Tool, ToolRegistry, @tool decorator (schema from type hints)
├── builtin_tools.py  # datetime / unit conversion / file read / FX rate
├── agent.py          # Agent: the ReAct loop, with optional memory
└── memory/
    ├── base.py           # Memory ABC (add / retrieve)
    ├── sqlite_memory.py  # episodic  — recent run summaries
    └── chroma_memory.py  # semantic  — facts by embedding similarity
examples/              # one runnable script per phase, plus demo.py
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate         # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then add your OPENAI_API_KEY
```

Or install the package itself:

```bash
pip install -e .                # core
pip install -e ".[semantic]"    # + ChromaDB for semantic memory
```

## Use it

```python
from agent_framework import Agent, ToolRegistry, SQLiteMemory

tools = ToolRegistry()

@tools.tool                      # schema is inferred from type hints + docstring
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: first number.
        b: second number.
    """
    return str(a + b)

agent = Agent(
    system_prompt="You are a helpful assistant.",
    registry=tools,
    episodic_memory=SQLiteMemory(),   # remembers across runs
)

print(agent.run("What is 21 + 21?"))
```

## Public API

| Name | Purpose |
| --- | --- |
| `Agent` | The ReAct loop. `run(user_input) -> str`, `reset()`. |
| `ToolRegistry` | Register / advertise / dispatch tools. `@registry.tool` decorator. |
| `tool`, `build_tool_from_function` | Build a tool schema from a function's type hints. |
| `ConversationManager` | Plain stateful chat, no tools (Phase 2). |
| `Memory` | Abstract base: `add(text, metadata)`, `retrieve(query, k)`. |
| `SQLiteMemory` | Episodic memory (recent summaries). |
| `ChromaMemory` | Semantic memory (vector similarity; needs `chromadb`). |

## Build order (and how to follow it)

Each phase is a commit and a runnable example. Read them in order:

| Phase | Idea | Run |
| --- | --- | --- |
| 1 | Raw stateless API call | `python examples/phase1_raw_call.py` |
| 2 | Multi-turn conversation state | `python examples/phase2_conversation.py` |
| 3 | Tool calling (delegation protocol) | `python examples/phase3_tools.py` |
| 4 | The agent loop | `python examples/phase4_agent.py` |
| 5 | Long-term memory | `python examples/phase5_memory.py` |
| 6 | Reusable package + demo | `python examples/demo.py` |

## The "am I done?" demo

`examples/demo.py` exercises all five completion criteria: general reasoning, an
external tool lookup, memory that survives a restart (`--recall`), and graceful
tool-failure handling. `examples/custom_agent.py` proves reusability by building
a *different* agent (a code reviewer) using only the public API.

```bash
python examples/demo.py
python examples/demo.py --recall     # memory persists across processes
python examples/custom_agent.py
```

## Design notes

- **The API is stateless.** History lives in *our* `messages` list and is resent
  every call. That single fact drives Phases 2–5.
- **The model never runs code.** It requests a tool call; `ToolRegistry.dispatch`
  runs it and returns a string — turning unknown tools, bad arguments, and tool
  exceptions into messages the agent can recover from instead of crashes.
- **`max_iterations`** caps a stuck agent. On reaching it, the agent asks for a
  best-effort answer with tools off rather than returning nothing.
- **Two memories, one interface.** Episodic (recency, SQLite) and semantic
  (similarity, Chroma) both implement the `Memory` ABC, so the Agent depends on
  the interface, not the backend.

## Cost

`gpt-4o-mini` ≈ $0.15 / 1M input tokens. A full agent run is usually < $0.01.
Budget a few dollars for the whole project.
