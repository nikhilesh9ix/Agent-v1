# Agent-v1

A small LLM **agent framework built from scratch** — the ReAct planning loop, a
tool-calling protocol, and layered memory — with no LangChain, CrewAI, or
LlamaIndex. An agent is a `while`-loop around an LLM wired to tools and memory;
this implements exactly that, and nothing you can't read in an afternoon.

Works with **OpenAI or Groq** (same OpenAI-compatible protocol), ships a
**Streamlit** chat UI, and comes with a test suite.

## Layout

```
agent_framework/
├── llm.py            # one shared client (chat + embeddings), provider switch
├── conversation.py   # ConversationManager: stateful multi-turn chat
├── tools.py          # Tool, ToolRegistry, @tool decorator (schema from type hints)
├── builtin_tools.py  # datetime / unit conversion / file read / FX rate
├── agent.py          # Agent: the ReAct loop, with optional memory
└── memory/
    ├── base.py           # Memory ABC (add / retrieve)
    ├── sqlite_memory.py  # episodic — recent run summaries
    └── chroma_memory.py  # semantic — facts by embedding similarity
app.py                # Streamlit chat frontend
examples/             # quickstart.py, custom_agent.py
tests/                # pytest suite (no network required)
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[semantic,app]"  # core + ChromaDB + Streamlit
cp .env.example .env               # then add your key(s)
```

### Providers

Groq and OpenAI both speak the OpenAI Chat Completions protocol, so one client
drives either — select with `LLM_PROVIDER` in `.env`:

```ini
LLM_PROVIDER=groq                   # or: openai
GROQ_API_KEY=gsk_...                # console.groq.com/keys
LLM_MODEL=llama-3.3-70b-versatile   # optional; sensible default per provider
```

Groq has no embeddings endpoint, so *semantic* (vector) memory always uses
OpenAI embeddings and needs `OPENAI_API_KEY` even under `LLM_PROVIDER=groq`.
Chat, tool calling, the agent loop, and *episodic* (SQLite) memory work on Groq
alone.

## Usage

```python
from agent_framework import Agent, ToolRegistry, SQLiteMemory

tools = ToolRegistry()

@tools.tool                      # schema inferred from type hints + docstring
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

Runnable examples: [`examples/quickstart.py`](examples/quickstart.py) (built-in
tools + memory) and [`examples/custom_agent.py`](examples/custom_agent.py) (a
code-review agent with custom tools).

```bash
python examples/quickstart.py
python examples/custom_agent.py
```

## Web UI

```bash
streamlit run app.py
```

Pick provider/model, edit the system prompt, toggle tools and memory, and watch
each turn's tool calls render inline.

## Public API

| Name | Purpose |
| --- | --- |
| `Agent` | The ReAct loop. `run(user_input) -> str`, `reset()`. |
| `ToolRegistry` | Register / advertise / dispatch tools; `@registry.tool`. |
| `tool`, `build_tool_from_function` | Build a tool schema from a function's type hints. |
| `ConversationManager` | Plain stateful chat, no tools. |
| `Memory` | Abstract base: `add(text, metadata)`, `retrieve(query, k)`. |
| `SQLiteMemory` | Episodic memory (recent summaries). |
| `ChromaMemory` | Semantic memory (vector similarity; needs `chromadb`). |

## Design notes

- **The API is stateless.** History lives in the `Agent`'s own `messages` list
  and is resent on every call.
- **The model never runs code.** It requests a tool call; `ToolRegistry.dispatch`
  runs it and returns a string — unknown tools, bad arguments, and tool
  exceptions all become messages the agent can recover from, not crashes.
- **`max_iterations`** caps a stuck agent. On reaching it, the agent asks for a
  best-effort answer with tools disabled rather than returning nothing.
- **Two backends, one interface.** Episodic (recency, SQLite) and semantic
  (similarity, Chroma) both implement the `Memory` ABC, so the `Agent` depends on
  the interface, not the backend.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
