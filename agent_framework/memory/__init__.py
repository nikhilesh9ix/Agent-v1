"""Long-term memory: SQLiteMemory (episodic) and ChromaMemory (semantic)."""

from .base import Memory
from .sqlite_memory import SQLiteMemory

__all__ = ["Memory", "SQLiteMemory"]

try:  # optional dependency
    from .chroma_memory import ChromaMemory  # noqa: F401
    __all__.append("ChromaMemory")
except ImportError:
    pass
