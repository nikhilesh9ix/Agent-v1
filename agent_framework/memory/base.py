"""The Memory interface: two backends (SQL, vector) behind one contract, so the
Agent depends on 'a memory', not on a specific store."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def add(self, text: str, metadata: dict | None = None) -> None:
        """Persist a piece of text (a summary, a fact) with optional metadata."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return up to k stored items relevant to `query` (recency or similarity)."""

    def close(self) -> None:
        """Release resources. Optional; default no-op."""
