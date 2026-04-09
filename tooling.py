"""Tool schemas and runtime executors for LLM tool-calling."""

from __future__ import annotations

from typing import Any

from google.genai import types

from session_store import SessionStore

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
                    "interesting labs or findings later."
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
        ]
    )
]


class ToolExecutor:
    def __init__(self, store: SessionStore) -> None:
        self._store = store

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
        """Persist a note to Redis under the user's session."""
        await self._store.save_research_note(session_id, note)
        return {"status": "saved", "session_id": session_id, "note": note}

    async def execute(self, tool_name: str, args: dict[str, Any]) -> Any:
        if tool_name == "search_research_labs":
            return self.search_research_labs(query=str(args.get("query", "")))
        if tool_name == "save_research_note":
            return await self.save_research_note(
                session_id=str(args.get("session_id", "")),
                note=str(args.get("note", "")),
            )
        raise ValueError(f"Unknown tool: {tool_name}")
