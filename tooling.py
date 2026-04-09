"""Tool schemas and runtime executors for LLM tool-calling."""

from __future__ import annotations

from typing import Any

from google.genai import types

from session_store import SessionStore
from vector_store import VectorStore

GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_research_labs",
                description=(
                    "Search for AI / ML research labs by name or topic. "
                    "Returns a list of matching labs with their research focus areas."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Lab name, research topic, or keyword to search for.",
                        )
                    },
                    required=["query"],
                ),
            ),
            types.FunctionDeclaration(
                name="save_research_note",
                description=(
                    "Save a research note for the active session so the user can revisit "
                    "interesting labs or findings later. Only call after user confirms."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "session_id": types.Schema(
                            type=types.Type.STRING,
                            description="Session id used to scope saved notes.",
                        ),
                        "note": types.Schema(
                            type=types.Type.STRING,
                            description="Short note to save.",
                        ),
                    },
                    required=["session_id", "note"],
                ),
            ),
            types.FunctionDeclaration(
                name="search_my_notes",
                description=(
                    "Semantically search the user's past research notes for this session. "
                    "Use this when the user asks about something they previously found "
                    "interesting or wants to recall earlier findings."
                ),
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "session_id": types.Schema(
                            type=types.Type.STRING,
                            description="Session id used to scope the note search.",
                        ),
                        "query": types.Schema(
                            type=types.Type.STRING,
                            description="Natural-language query to find relevant past notes.",
                        ),
                    },
                    required=["session_id", "query"],
                ),
            ),
        ]
    )
]


class ToolExecutor:
    def __init__(self, store: SessionStore, vector_store: VectorStore) -> None:
        self._store = store
        self._vector_store = vector_store

    @staticmethod
    def search_research_labs(query: str) -> list[dict[str, Any]]:
        """Return a dummy list of AI research labs whose name or topics match *query*."""
        labs = [
            {"name": "DeepMind", "topics": ["reinforcement learning", "protein folding", "AlphaCode"]},
            {"name": "Anthropic", "topics": ["AI safety", "constitutional AI", "interpretability"]},
            {"name": "OpenAI", "topics": ["GPT", "DALL-E", "Sora", "reasoning"]},
            {"name": "MIT CSAIL", "topics": ["robotics", "natural language processing", "computer vision"]},
            {"name": "Stanford HAI", "topics": ["human-centered AI", "policy", "ethics"]},
            {"name": "Google Brain / Google DeepMind", "topics": ["transformers", "diffusion models", "Gemini"]},
            {"name": "CMU LTI", "topics": ["NLP", "dialogue systems", "multimodal AI"]},
        ]
        q = query.lower()
        results = [
            lab for lab in labs if q in lab["name"].lower() or any(q in t for t in lab["topics"])
        ]
        return results or labs

    async def save_research_note(self, session_id: str, note: str) -> dict[str, str]:
        """Persist a note to both Redis (fast list) and the vector store (semantic search)."""
        await self._store.save_research_note(session_id, note)
        await self._vector_store.add_note(session_id, note)
        return {"status": "saved", "session_id": session_id, "note": note}

    async def search_my_notes(self, session_id: str, query: str) -> dict[str, Any]:
        """Return the top-3 most semantically relevant past notes for *query*."""
        hits = await self._vector_store.search_notes(session_id, query)
        return {"results": hits, "count": len(hits)}

    async def execute(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "search_research_labs":
            return self.search_research_labs(query=str(args.get("query", "")))
        if tool_name == "save_research_note":
            return await self.save_research_note(
                session_id=str(args.get("session_id", "")),
                note=str(args.get("note", "")),
            )
        if tool_name == "search_my_notes":
            return await self.search_my_notes(
                session_id=str(args.get("session_id", "")),
                query=str(args.get("query", "")),
            )
        raise ValueError(f"Unknown tool: {tool_name}")
