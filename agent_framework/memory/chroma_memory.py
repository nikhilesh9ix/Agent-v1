"""Semantic memory backed by ChromaDB.

Facts are stored as embedding vectors. Retrieval embeds the incoming query and
returns the nearest stored facts by cosine similarity — so "the user likes dark
interfaces" can surface for a query about "dark mode" even with no shared words.

We compute embeddings ourselves via the shared llm.embed() so the same model and
key are used everywhere, and Chroma is just the vector index.
"""

from __future__ import annotations

import uuid

import chromadb

from .. import llm
from .base import Memory


class ChromaMemory(Memory):
    """Meaning-based recall over a persistent local vector store."""

    def __init__(self, collection_name: str = "agent_semantic",
                 persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # We pass our own embeddings, so no server-side embedding function.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Embed a fact and store it with an auto-generated id."""
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[llm.embed(text)],
            metadatas=[metadata or {"source": "agent"}],
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the k stored facts most semantically similar to `query`."""
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[llm.embed(query)],
            n_results=min(k, count),
        )
        # query() returns lists-of-lists (one row per query embedding).
        return result["documents"][0] if result["documents"] else []
