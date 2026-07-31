"""Episodic memory: run summaries in a SQLite file, retrieved by recency."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .base import Memory


class SQLiteMemory(Memory):
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "user_query TEXT, summary TEXT NOT NULL)")
        self.conn.commit()

    def add(self, text: str, metadata: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO episodes (timestamp, user_query, summary) VALUES (?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), (metadata or {}).get("user_query", ""), text))
        self.conn.commit()

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        """The k most recent summaries, oldest-first (query unused — recency wins)."""
        rows = self.conn.execute(
            "SELECT timestamp, user_query, summary FROM episodes ORDER BY id DESC LIMIT ?",
            (k,)).fetchall()
        rows.reverse()
        return [f"[{ts}] (asked: {q}) {s}" if q else f"[{ts}] {s}" for ts, q, s in rows]

    def close(self) -> None:
        self.conn.close()
