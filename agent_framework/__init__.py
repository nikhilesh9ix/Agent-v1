"""agent_framework — a from-scratch LLM agent framework.

Built without LangChain/CrewAI/etc. to expose exactly what those libraries do
under the hood: the ReAct loop, tool calling, and layered memory.

Quick start
-----------
    from agent_framework import Agent, ToolRegistry, SQLiteMemory

    tools = ToolRegistry()

    @tools.tool
    def add(a: int, b: int) -> str:
        '''Add two integers.'''
        return str(a + b)

    agent = Agent("You are a helpful assistant.", registry=tools,
                  episodic_memory=SQLiteMemory())
    print(agent.run("What is 21 + 21?"))
"""

from .agent import Agent
from .conversation import ConversationManager
from .memory import Memory, SQLiteMemory
from .tools import Tool, ToolRegistry, build_tool_from_function, tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "ConversationManager",
    "ToolRegistry",
    "Tool",
    "tool",
    "build_tool_from_function",
    "Memory",
    "SQLiteMemory",
    "__version__",
]

# Semantic memory needs the optional chromadb dependency.
try:  # pragma: no cover - optional dependency
    from .memory import ChromaMemory  # noqa: F401
    __all__.append("ChromaMemory")
except ImportError:  # pragma: no cover
    pass
