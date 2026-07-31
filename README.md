# Agent-v1 — an agent framework from scratch

Build the planning loop, memory system, and tool-calling protocol of an LLM
agent **without** LangChain, CrewAI, or any agent library — so you understand
exactly what those frameworks do under the hood.

> Status: work in progress, built phase by phase. See commit history.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
copy .env.example .env      # then add your OPENAI_API_KEY
```

## Build order

| Phase | What | Where |
| --- | --- | --- |
| 1 | Raw LLM API call | `examples/phase1_raw_call.py` |
| 2 | ConversationManager (multi-turn) | `agent_framework/conversation.py` |
| 3 | Tool calling | `agent_framework/tools.py` |
| 4 | Agent loop (ReAct) | `agent_framework/agent.py` |
| 5 | Memory (SQLite + ChromaDB) | `agent_framework/memory/` |
| 6 | Reusable package + demo | package root + `examples/demo.py` |

## Cost

`gpt-4o-mini` ~= $0.15 / 1M input tokens. A full run is usually < $0.01.
