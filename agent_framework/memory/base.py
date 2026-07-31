"""The Memory interface.

Two very different backends (a SQL table, a vector store) hide behind the same
two operations. Naming the minimal contract as an abstract base class is what
lets the Agent depend on "a memory" rather than on SQLite or Chroma specifically
— swap the implementation, the Agent doesn't change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Memory(ABC):
    """Minimal long-term memory contract: write something, read back the relevant bits."""

    @abstractmethod
    def add(self, text: str, metadata: dict | None = None) -> None:
        """Persist a piece of text (a summary, a fact) with optional metadata."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return up to k stored items relevant to `query`, as strings.

        "Relevant" is backend-defined: recency for episodic storage, semantic
        similarity for a vector store.
        """

    def close(self) -> None:
        """Release resources (connections, files). Optional; default no-op."""
