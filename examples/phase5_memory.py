"""Phase 5 demo: an agent that remembers across separate run() calls.

    python examples/phase5_memory.py          # first time: teach it a fact
    python examples/phase5_memory.py --recall  # later run: it recalls the fact

Because memory is persisted to disk (SQLite file + Chroma dir), the second
invocation is a *fresh process* yet still recalls what the first one learned —
proving the memory outlives the program, not just the loop.
"""

import sys

from agent_framework.agent import Agent
from agent_framework.builtin_tools import default_registry
from agent_framework.memory import SQLiteMemory

try:
    from agent_framework.memory import ChromaMemory
    semantic = ChromaMemory()
except ImportError:
    print("[chromadb not installed — running with episodic memory only]")
    semantic = None


def build_agent() -> Agent:
    return Agent(
        system_prompt="You are a helpful personal assistant with a good memory.",
        registry=default_registry(),
        episodic_memory=SQLiteMemory(),
        semantic_memory=semantic,
        verbose=True,
    )


def main() -> None:
    agent = build_agent()
    if "--recall" in sys.argv:
        q = "What do you remember about my preferences?"
    else:
        q = "Remember: I prefer dark mode and I'm allergic to peanuts."
    print(f"\nUSER: {q}")
    print("AGENT:", agent.run(q))


if __name__ == "__main__":
    main()
