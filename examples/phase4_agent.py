"""Phase 4 demo: the agent runs the tool loop on its own.

    python examples/phase4_agent.py

Ask a question that needs a tool (or several) and watch the agent reason, call
tools, observe results, and answer — no manual round-trip wiring.
"""

from agent_framework.agent import Agent
from agent_framework.builtin_tools import default_registry


def main() -> None:
    agent = Agent(
        system_prompt=(
            "You are a helpful assistant. Use the available tools when they help "
            "you answer accurately. Think step by step."
        ),
        registry=default_registry(),
        verbose=True,
    )

    for question in [
        "What is 12 miles in kilometers, and what is the current UTC time?",
        "What is 1 USD in EUR right now?",
    ]:
        print(f"\nUSER: {question}")
        print("AGENT:", agent.run(question))
        agent.reset()


if __name__ == "__main__":
    main()
