import asyncio
import json
import os
import time
from typing import Any, Callable, Final

import httpx
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from logging_setup import (
    log_llm_chat_completion,
    log_rolling_summary_refresh_complete,
    log_rolling_summary_refresh_failed,
    log_session_reset,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from schemas import ChatRequest, ChatResponse
from session_store import Exchange, SessionStore
from tracing import (
    get_client,
    otel_attached,
    otel_context_current,
    propagate_attributes,
    usage_details_for_langfuse,
)

_CHAT_SYSTEM_PROMPT: Final[str] = (
    "You are an expert research assistant. Use the provided conversation summary "
    "for prior context and answer the user's latest message concisely. You have "
    "access to a tool named search_research_labs that can fetch lab names and "
    "research topics. Be proactive: if the user asks about research topics, "
    "specific labs, or factual information about academic/research institutions, "
    "use search_research_labs instead of relying only on internal knowledge. "
    "When the question does not require tool data, answer directly. "
    "When faced with ambiguous requests such as 'interesting projects' or "
    "'what should I explore', use the conversation summary and the user's "
    "professional background to suggest 3 specific research areas. "
    "Only call search_research_labs once the user has confirmed a focus area."
)

_SUMMARY_UPDATE_SYSTEM_PROMPT: Final[str] = (
    "You maintain a rolling summary of a conversation. You will be given the "
    "previous summary and a batch of recent exchanges. Perform inductive "
    "reasoning: identify what new facts, decisions, or context must be "
    "preserved, merge them with the old summary, and return an updated summary "
    "under 200 words. Output plain text only—no preamble, headings, or markdown."
)

_BUFFER_SIZE: Final[int] = 3  # exchanges before a summary flush is triggered

_USER_CONTEXT: Final[str] = "Data Analyst based in Tel Aviv"

_MAX_TOOL_ITERATIONS: Final[int] = 5  # guard against infinite tool loops


class InvalidLLMResponseError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Tool definitions — add new tools here; register them in _TOOL_REGISTRY
# ---------------------------------------------------------------------------

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
        lab for lab in labs
        if q in lab["name"].lower() or any(q in t for t in lab["topics"])
    ]
    return results or labs  # fall back to full list if nothing matches


_TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "search_research_labs": search_research_labs,
}

_GEMINI_TOOLS = [
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
            )
        ]
    )
]


# ---------------------------------------------------------------------------
# Gemini backend — the only provider-specific section
# ---------------------------------------------------------------------------

def _is_retryable(exc: BaseException) -> bool:
    """Network errors and rate-limit responses are transient and worth retrying."""
    if isinstance(exc, httpx.RequestError):
        return True
    return isinstance(exc, ClientError) and exc.code == 429


async def _gemini_call_raw(
    client: genai.Client,
    model: str,
    messages: list[dict[str, str]],
) -> tuple[str, object]:
    """Single raw Gemini API call, no retry logic."""
    response = await client.aio.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if content is None or not content.strip():
        raise InvalidLLMResponseError("LLM returned empty content")
    return content, response


# User-facing: fail fast — 3 attempts, max 10 s wait.
_gemini_call_interactive = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)(_gemini_call_raw)

# Background: prioritise eventual success — 8 attempts, up to 5 min wait.
_gemini_call_background = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(8),
    wait=wait_exponential(multiplier=2, min=2, max=300),
    reraise=True,
)(_gemini_call_raw)


class _GeminiClient:
    """Thin wrapper that owns the Gemini SDK client and exposes a provider-agnostic call."""

    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("GEMINI_MODEL")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if not model:
            raise RuntimeError("GEMINI_MODEL is not set")
        self.model = model
        self._client = genai.Client(api_key=api_key)

    async def complete(self, messages: list[dict[str, str]]) -> tuple[str, object]:
        return await _gemini_call_interactive(self._client, self.model, messages)

    async def complete_background(self, messages: list[dict[str, str]]) -> tuple[str, object]:
        return await _gemini_call_background(self._client, self.model, messages)

    @staticmethod
    def _build_contents(user_message: str) -> list[types.Content]:
        """Wrap a plain user string into the native Gemini Content format."""
        return [types.Content(role="user", parts=[types.Part.from_text(text=user_message)])]

    @staticmethod
    def _execute_tool_calls(
        function_calls: list[Any],
    ) -> list[types.Part]:
        """Run every tool requested by the model and return the result parts."""
        result_parts: list[types.Part] = []
        for fc in function_calls:
            fn = _TOOL_REGISTRY.get(fc.name)
            tool_output: Any = fn(**dict(fc.args)) if fn else {"error": f"Unknown tool: {fc.name}"}
            result_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.dumps(tool_output)},
                )
            )
        return result_parts

    async def _react_loop(
        self,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
    ) -> object:
        """Drive the ReAct loop: call → tool-execute → repeat until model stops or cap hit."""
        last_response: object = None
        cap_reached = True
        for _ in range(_MAX_TOOL_ITERATIONS):
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config,
            )
            last_response = response

            function_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if part.function_call is not None
            ]
            if not function_calls:
                cap_reached = False
                break

            contents.append(response.candidates[0].content)
            contents.append(
                types.Content(role="user", parts=self._execute_tool_calls(function_calls))
            )

        if cap_reached:
            import logging
            logging.getLogger("llm").warning(
                "tool_loop_cap_reached", extra={"max_iterations": _MAX_TOOL_ITERATIONS}
            )
        return last_response

    async def complete_with_tools(
        self,
        user_message: str,
        system_prompt: str,
    ) -> tuple[str, object]:
        """Native Gemini API call with an automatic ReAct tool-use loop."""
        contents = self._build_contents(user_message)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt, tools=_GEMINI_TOOLS
        )
        last_response = await self._react_loop(contents, config)
        text = getattr(last_response, "text", None)
        if not text or not text.strip():
            raise InvalidLLMResponseError("LLM returned empty content after tool loop")
        return text.strip(), last_response

    async def ping(self) -> None:
        await self._client.aio.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            extra_body={"max_completion_tokens": 1},
        )


# ---------------------------------------------------------------------------
# Provider-agnostic layer — session management, rolling summary, tracing
# ---------------------------------------------------------------------------

class LLMProvider:
    def __init__(self, store: SessionStore) -> None:
        self._backend = _GeminiClient()
        self._store = store
        self.model = self._backend.model

    def _schedule_background(
        self, session_id: str, snapshot_summary: str, snapshot_buffer: list[Exchange]
    ) -> None:
        """Fire-and-forget the summary refresh, keeping it on the current trace."""
        ctx = otel_context_current()
        asyncio.create_task(
            self._run_in_context(
                ctx,
                self._background_refresh_summary(session_id, snapshot_summary, snapshot_buffer),
            )
        )

    async def _chat_completion_text(
        self,
        messages: list[dict[str, str]],
        *,
        observation_name: str = "llm-completion",
        background: bool = False,
    ) -> str:
        """OpenAI-compat path — used for background summary refresh (no tools)."""
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="generation",
            name=observation_name,
            model=self.model,
        ) as gen:
            gen.update(input=messages)
            call = self._backend.complete_background if background else self._backend.complete
            content, response = await call(messages)
            usage = usage_details_for_langfuse(response)
            gen.update(output=content, usage_details=usage or {})
            log_llm_chat_completion(observation_name, usage)
            return content

    async def _chat_with_tools(self, user_message: str) -> str:
        """Native Gemini path — used for interactive chat, supports tool calls."""
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-chat-reply",
            model=self.model,
        ) as gen:
            gen.update(input={"user_message": user_message})
            content, response = await self._backend.complete_with_tools(
                user_message, _CHAT_SYSTEM_PROMPT
            )
            usage = usage_details_for_langfuse(response)
            gen.update(output=content, usage_details=usage or {})
            log_llm_chat_completion("llm-chat-reply", usage)
            return content

    async def update_rolling_summary(
        self,
        current_summary: str,
        buffer: list[Exchange],
    ) -> str:
        exchanges_text = "\n\n".join(
            f"User: {ex.user}\nAssistant: {ex.assistant}" for ex in buffer
        )
        user_content = (
            f"Previous summary:\n{current_summary.strip() or '(none yet)'}\n\n"
            f"Recent exchanges ({len(buffer)}):\n{exchanges_text}\n\n"
            "Updated summary (under 200 words):"
        )
        return await self._chat_completion_text(
            [
                {"role": "system", "content": _SUMMARY_UPDATE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            observation_name="llm-summary-refresh",
            background=True,
        )

    async def _background_refresh_summary(
        self,
        session_id: str,
        snapshot_summary: str,
        snapshot_buffer: list[Exchange],
    ) -> None:
        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name="rolling-summary-update") as span:
            span.update(input={
                "session_id": session_id,
                "buffer_size": len(snapshot_buffer),
                "exchanges_preview": [
                    {"user": ex.user[:200], "assistant": ex.assistant[:200]}
                    for ex in snapshot_buffer
                ],
            })
            t0 = time.perf_counter()
            try:
                updated = await self.update_rolling_summary(snapshot_summary, snapshot_buffer)
            except Exception:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                span.update(metadata={"summary_refresh_latency_ms": elapsed_ms, "summary_refresh_status": "error"})
                log_rolling_summary_refresh_failed(session_id, elapsed_ms)
                return

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.update(metadata={"summary_refresh_latency_ms": elapsed_ms, "summary_refresh_status": "ok"})
            log_rolling_summary_refresh_complete(session_id, elapsed_ms)

            if updated.strip():
                text = updated.strip()
                async with self._store.session_lock(session_id):
                    state = await self._store.get_state(session_id)
                    state.summary = text
                    # Trim only the exchanges that were included in this flush.
                    state.buffer = state.buffer[len(snapshot_buffer):]
                    await self._store.set_state(session_id, state)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        sid = request.session_id
        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name="chat-turn"):
            with propagate_attributes(session_id=sid):
                async with self._store.session_lock(sid):
                    state = await self._store.get_state(sid)
                    summary, buffer_len = state.summary, len(state.buffer)

                context_block = f"User background: {_USER_CONTEXT}"
                summary_block = (
                    f"Conversation summary so far:\n{summary.strip()}"
                    if summary.strip() else ""
                )
                preamble = "\n\n".join(filter(None, [context_block, summary_block]))
                user_turn = f"{preamble}\n\nUser message:\n{request.message}"
                langfuse.update_current_span(input={
                    "user_message": request.message,
                    "had_prior_summary": bool(summary.strip()),
                    "buffer_len": buffer_len,
                })

                reply = await self._chat_with_tools(user_turn)
                langfuse.update_current_span(output={"reply": reply})

                async with self._store.session_lock(sid):
                    state = await self._store.get_state(sid)
                    state.buffer.append(Exchange(user=request.message, assistant=reply))
                    await self._store.set_state(sid, state)
                    if len(state.buffer) >= _BUFFER_SIZE:
                        self._schedule_background(sid, state.summary, list[Exchange](state.buffer))

                return ChatResponse(session_id=sid, reply=reply)

    @staticmethod
    async def _run_in_context(ctx, coro) -> None:
        with otel_attached(ctx):
            await coro

    async def reset_session(self, session_id: str) -> bool:
        """Soft-delete session data. Returns True if a session existed."""
        existed = await self._store.soft_delete_state(session_id)
        log_session_reset(session_id, existed=existed)
        return existed

    async def check_readiness(self) -> None:
        await self._backend.ping()
        await self._store.ping()
