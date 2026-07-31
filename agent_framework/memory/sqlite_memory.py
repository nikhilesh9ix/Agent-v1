"""Episodic memory backed by SQLite.

Stores a short summary of each run with a timestamp and the original query. On
retrieval it returns the most *recent* episodes — the agent's answer to "what
were we just doing?". No server, one file on disk: persistence with zero infra.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .base import Memory


class SQLiteMemory(Memory):
    """Recency-ordered episodic memory in a single SQLite file."""

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        # check_same_thread=False keeps it usable from simple scripts/tests.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                user_query TEXT,
                summary    TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Store one episode. metadata may carry {'user_query': ...}."""
        metadata = metadata or {}
        self.conn.execute(
            "INSERT INTO episodes (timestamp, user_query, summary) VALUES (?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                metadata.get("user_query", ""),
                text,
            ),
        )
        self.conn.commit()

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """Return the k most recent episode summaries (query is unused here).

        Episodic recall is about recency, not similarity — the newest episodes
        are the relevant ones. Returned oldest-first so they read chronologically.
        """
        rows = self.conn.execute(
            "SELECT timestamp, user_query, summary FROM episodes ORDER BY id DESC LIMIT ?",
            (k,),
        ).fetchall()
        rows.reverse()
        return [
            f"[{ts}] (asked: {q}) {summary}" if q else f"[{ts}] {summary}"
            for ts, q, summary in rows
        ]

    def close(self) -> None:
        self.conn.close()
