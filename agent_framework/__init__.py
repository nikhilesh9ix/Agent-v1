"""agent_framework — a small LLM agent framework: ReAct loop and tool calling."""

from .agent import Agent
from .conversation import ConversationManager
from .tools import Tool, ToolRegistry, build_tool_from_function, tool

__version__ = "0.1.0"

__all__ = ["Agent", "ConversationManager", "ToolRegistry", "Tool", "tool",
           "build_tool_from_function", "__version__"]
