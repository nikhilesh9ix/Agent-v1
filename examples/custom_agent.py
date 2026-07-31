"""Proof of reusability: a *different* agent, using only the public API.

A code-review assistant with its own custom tools — registered via the @tool
decorator with schemas inferred from type hints, no framework internals touched.
This is criterion 5: someone can build a new agent from the public interface.

    python examples/custom_agent.py
"""

from agent_framework import Agent, ToolRegistry

tools = ToolRegistry()


@tools.tool
def count_lines(code: str) -> str:
    """Count the number of lines in a code snippet.

    Args:
        code: The source code to measure.
    """
    return str(len(code.splitlines()))


@tools.tool
def find_todos(code: str) -> str:
    """Find TODO/FIXME markers in a code snippet and report their line numbers.

    Args:
        code: The source code to scan.
    """
    hits = [
        f"line {i}: {line.strip()}"
        for i, line in enumerate(code.splitlines(), 1)
        if "TODO" in line or "FIXME" in line
    ]
    return "\n".join(hits) if hits else "No TODO/FIXME markers found."


def main() -> None:
    reviewer = Agent(
        system_prompt=(
            "You are a senior code reviewer. Use the tools to gather facts about "
            "the snippet, then give concise, actionable feedback."
        ),
        registry=tools,
        verbose=True,
    )

    snippet = (
        "def add(a, b):\n"
        "    # TODO: validate inputs\n"
        "    return a + b\n"
        "\n"
        "def divide(a, b):\n"
        "    return a / b  # FIXME: no zero check\n"
    )

    print("AGENT:", reviewer.run(
        f"Review this snippet. How many lines, any TODOs/FIXMEs, and what should I fix?\n\n{snippet}"
    ))


if __name__ == "__main__":
    main()
