"""Semantic memory: facts stored as embedding vectors in ChromaDB, retrieved by
meaning similarity. Embeddings come from llm.embed() (OpenAI)."""

from __future__ import annotations

import uuid

import chromadb

from .. import llm
from .base import Memory


class ChromaMemory(Memory):
    def __init__(self, collection_name: str = "agent_semantic", persist_dir: str = "chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

    def add(self, text: str, metadata: dict | None = None) -> None:
        self.collection.add(ids=[str(uuid.uuid4())], documents=[text],
                            embeddings=[llm.embed(text)], metadatas=[metadata or {"source": "agent"}])

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(query_embeddings=[llm.embed(query)], n_results=min(k, count))
        return result["documents"][0] if result["documents"] else []
