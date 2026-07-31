"""Memory backends for the agent.

Short-term memory is the message list the model already sees. This subpackage is
*long-term* memory — storage that outlives a single run() and is selectively
pulled back into context:

- SQLiteMemory  : episodic  — "what happened before" (recent run summaries).
- ChromaMemory  : semantic  — "what do I know" (facts retrieved by meaning).

Both implement the same tiny Memory interface (add / retrieve), so the Agent
treats them interchangeably.
"""

from .base import Memory
from .sqlite_memory import SQLiteMemory

__all__ = ["Memory", "SQLiteMemory"]

# ChromaMemory needs the optional `chromadb` dependency; import lazily so the
# rest of the framework works without it installed.
try:  # pragma: no cover - optional dependency
    from .chroma_memory import ChromaMemory  # noqa: F401
    __all__.append("ChromaMemory")
except ImportError:  # pragma: no cover
    pass
