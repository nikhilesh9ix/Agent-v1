"""agent_framework — a small LLM agent framework: ReAct loop, tool calling, memory."""

from .agent import Agent
from .conversation import ConversationManager
from .memory import Memory, SQLiteMemory
from .tools import Tool, ToolRegistry, build_tool_from_function, tool

__version__ = "0.1.0"

__all__ = ["Agent", "ConversationManager", "ToolRegistry", "Tool", "tool",
           "build_tool_from_function", "Memory", "SQLiteMemory", "__version__"]

try:  # ChromaMemory needs the optional chromadb dependency
    from .memory import ChromaMemory  # noqa: F401
    __all__.append("ChromaMemory")
except ImportError:
    pass
