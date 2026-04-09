"""Vector store for semantic note search using Gemini embeddings + ChromaDB."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from uuid import uuid4

import chromadb
from chromadb import Collection
from google import genai

_EMBED_MODEL: str = "text-embedding-004"
_TOP_K_DEFAULT: int = 3


class VectorStore:
    """Persist research notes as embedding vectors and retrieve by semantic similarity.

    Uses ChromaDB for ANN search and the Gemini Embeddings API to embed text.
    Each note is scoped to a session via ChromaDB metadata filtering so users
    never see each other's notes.
    """

    def __init__(self, collection: Collection, genai_client: genai.Client) -> None:
        self._collection = collection
        self._genai = genai_client

    @classmethod
    def from_env(cls) -> "VectorStore":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        persist_dir = os.getenv("CHROMA_PERSIST_DIR")
        chroma = (
            chromadb.PersistentClient(path=persist_dir)
            if persist_dir
            else chromadb.EphemeralClient()
        )
        collection = chroma.get_or_create_collection(
            name="research_notes",
            # cosine similarity is best for comparing text embeddings.
            metadata={"hnsw:space": "cosine"},
        )
        return cls(collection, genai.Client(api_key=api_key))

    @classmethod
    def ephemeral(cls, genai_client: genai.Client) -> "VectorStore":
        """In-memory instance for tests — no disk I/O."""
        chroma = chromadb.EphemeralClient()
        collection = chroma.get_or_create_collection(
            name="research_notes",
            metadata={"hnsw:space": "cosine"},
        )
        return cls(collection, genai_client)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        result = await self._genai.aio.models.embed_content(
            model=_EMBED_MODEL,
            contents=text,
        )
        return list(result.embeddings[0].values)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_note(self, session_id: str, note: str) -> None:
        """Embed *note* and upsert it into the collection scoped to *session_id*."""
        vector = await self._embed(note)
        note_id = str(uuid4())
        # ChromaDB is synchronous — run in a thread to avoid blocking the event loop.
        await asyncio.to_thread(
            self._collection.add,
            ids=[note_id],
            embeddings=[vector],
            documents=[note],
            metadatas=[{"session_id": session_id}],
        )

    async def search_notes(
        self, session_id: str, query: str, top_k: int = _TOP_K_DEFAULT
    ) -> list[dict[str, Any]]:
        """Return the *top_k* most semantically relevant notes for *query*.

        Results are filtered to the caller's *session_id* so cross-session
        leakage is impossible at the storage layer.
        """
        query_vector = await self._embed(query)
        results = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"session_id": session_id},
            include=["documents", "distances"],
        )
        ids = results.get("ids", [[]])[0]
        notes = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        # Convert cosine distance → similarity score (0–1, higher = more relevant).
        return [
            {"id": note_id, "content": note, "similarity": round(1 - dist, 4)}
            for note_id, note, dist in zip(ids, notes, distances)
        ]
