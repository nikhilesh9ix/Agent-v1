"""End-to-end demo built ONLY on agent_framework (no LangChain/AutoGen).

Exercises the five "how to know you're done" criteria:
  1. Answer a general question from its own reasoning (no tool).
  2. Look up current information via an external tool.
  3. Remember facts across separate run() sessions (persisted to disk).
  4. Handle a tool failure gracefully (explain, not crash).
  5. (see examples/custom_agent.py) build a *different* agent with the same API.

    python examples/demo.py
    python examples/demo.py --recall   # prove memory survives a restart
"""

import sys

from agent_framework import Agent, SQLiteMemory
from agent_framework.builtin_tools import default_registry


def build_agent() -> Agent:
    return Agent(
        system_prompt=(
            "You are a capable assistant. Use tools when they improve accuracy. "
            "If a tool fails, explain what went wrong and continue helpfully."
        ),
        registry=default_registry(),
        episodic_memory=SQLiteMemory("demo_memory.db"),
        verbose=True,
    )


def ask(agent: Agent, label: str, question: str) -> None:
    print(f"\n=== {label} ===\nUSER: {question}")
    print("AGENT:", agent.run(question))
    agent.reset()


def main() -> None:
    agent = build_agent()

    if "--recall" in sys.argv:
        # Second process: memory from the previous run should resurface.
        ask(agent, "3. Cross-session memory (recall)",
            "Based on what you remember, what did we discuss and what do I like?")
        return

    ask(agent, "1. General reasoning (no tool)",
        "In one sentence, what is the difference between an agent and a plain chatbot?")

    ask(agent, "2. External tool lookup",
        "What is the current exchange rate from USD to INR?")

    ask(agent, "4. Graceful tool failure",
        "Read the file ./definitely_missing_file.txt and tell me what happened.")

    ask(agent, "3. Memory (teach a fact)",
        "Remember that I prefer metric units and I'm learning to build AI agents.")

    print("\nNow rerun with:  python examples/demo.py --recall")


if __name__ == "__main__":
    main()
