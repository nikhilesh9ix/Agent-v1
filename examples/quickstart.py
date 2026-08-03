"""Minimal end-to-end usage of the framework.

Builds an agent with the built-in tools, then asks questions that require tool
use. Run with the package installed (`pip install -e .`) and a provider key in
`.env`:

    python examples/quickstart.py
"""

from agent_framework import Agent
from agent_framework.builtin_tools import default_registry


def main() -> None:
    agent = Agent(
        system_prompt="You are a helpful assistant. Use tools when they improve accuracy.",
        registry=default_registry(),
    )
    for question in (
        "What is 12 miles in kilometers?",
        "What is the current exchange rate from USD to INR?",
    ):
        print(f"\nQ: {question}")
        print(f"A: {agent.run(question)}")
        agent.reset()


if __name__ == "__main__":
    main()
