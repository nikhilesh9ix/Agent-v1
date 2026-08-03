# Agent-v1

A small LLM **agent framework built from scratch** — the ReAct planning loop and
a tool-calling protocol — with no LangChain, CrewAI, or LlamaIndex. An agent is a
`while`-loop around an LLM wired to tools; this implements exactly that, and
nothing you can't read in an afternoon.

Works with **OpenAI or Groq** (same OpenAI-compatible protocol) and ships a
**Streamlit** chat UI plus a test suite.

## Layout

```
agent_framework/
├── llm.py            # one shared client (chat), provider switch
├── conversation.py   # ConversationManager: stateful multi-turn chat
├── tools.py          # Tool, ToolRegistry, @tool decorator (schema from type hints)
├── builtin_tools.py  # datetime / unit conversion / file read / FX rate
└── agent.py          # Agent: the ReAct loop
app.py                # Streamlit chat frontend
examples/             # quickstart.py, custom_agent.py
tests/                # pytest suite (no network required)
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows  (source .venv/bin/activate elsewhere)
pip install -e ".[app]"    # core + Streamlit
cp .env.example .env         # then add your key
```

### Providers

Groq and OpenAI both speak the OpenAI Chat Completions protocol, so one client
drives either — select with `LLM_PROVIDER` in `.env`:

```ini
LLM_PROVIDER=groq                   # or: openai
GROQ_API_KEY=gsk_...                # console.groq.com/keys
LLM_MODEL=llama-3.3-70b-versatile   # optional; sensible default per provider
```

## Usage

```python
from agent_framework import Agent, ToolRegistry

tools = ToolRegistry()

@tools.tool                      # schema inferred from type hints + docstring
def add(a: int, b: int) -> str:
    """Add two integers.

    Args:
        a: first number.
        b: second number.
    """
    return str(a + b)

agent = Agent("You are a helpful assistant.", registry=tools)
print(agent.run("What is 21 + 21?"))
```

Runnable examples: [`examples/quickstart.py`](examples/quickstart.py) (built-in
tools) and [`examples/custom_agent.py`](examples/custom_agent.py) (a code-review
agent with custom tools).

```bash
python examples/quickstart.py
python examples/custom_agent.py
```

## Web UI

```bash
streamlit run app.py
```

Pick provider/model, edit the system prompt, toggle tools, and watch each turn's
tool calls render inline.

## Public API

| Name | Purpose |
| --- | --- |
| `Agent` | The ReAct loop. `run(user_input) -> str`, `reset()`. |
| `ToolRegistry` | Register / advertise / dispatch tools; `@registry.tool`. |
| `tool`, `build_tool_from_function` | Build a tool schema from a function's type hints. |
| `ConversationManager` | Plain stateful chat, no tools. |

## Design notes

- **The API is stateless.** History lives in the `Agent`'s own `messages` list
  and is resent on every call.
- **The model never runs code.** It requests a tool call; `ToolRegistry.dispatch`
  runs it and returns a string — unknown tools, bad arguments, and tool
  exceptions all become messages the agent can recover from, not crashes.
- **`max_iterations`** caps a stuck agent. On reaching it, the agent asks for a
  best-effort answer with tools disabled rather than returning nothing.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
