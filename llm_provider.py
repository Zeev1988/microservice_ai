import asyncio
import os
import time
from typing import Final

import httpx
from google import genai
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
    "You are a helpful assistant. Use the provided conversation summary for "
    "prior context. Answer the user's latest message concisely."
)

_SUMMARY_UPDATE_SYSTEM_PROMPT: Final[str] = (
    "You maintain a rolling summary of a conversation. You will be given the "
    "previous summary and a batch of recent exchanges. Perform inductive "
    "reasoning: identify what new facts, decisions, or context must be "
    "preserved, merge them with the old summary, and return an updated summary "
    "under 200 words. Output plain text only—no preamble, headings, or markdown."
)

_BUFFER_SIZE: Final[int] = 3  # exchanges before a summary flush is triggered


class InvalidLLMResponseError(ValueError):
    pass


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

                user_turn = (
                    f"Conversation summary so far:\n{summary.strip()}\n\n"
                    f"User message:\n{request.message}"
                    if summary.strip() else request.message
                )
                langfuse.update_current_span(input={
                    "user_message": request.message,
                    "had_prior_summary": bool(summary.strip()),
                    "buffer_len": buffer_len,
                })

                reply = await self._chat_completion_text(
                    [
                        {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_turn},
                    ],
                    observation_name="llm-chat-reply",
                )
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
