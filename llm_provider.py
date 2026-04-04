import asyncio
import os
import time
from typing import Final, Protocol, runtime_checkable

import httpx
from google import genai
from google.genai.errors import ClientError
from logging_setup import (
    log_llm_chat_completion,
    log_rolling_summary_refresh_complete,
    log_rolling_summary_refresh_failed,
)
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from schemas import ChatRequest, ChatResponse
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
    "You maintain a rolling summary of a conversation. Given the previous "
    "summary and the latest user/assistant exchange, produce an updated "
    "summary that incorporates the new information. Keep the result under "
    "200 words. Output plain text only—no preamble, headings, or markdown."
)

_SESSION_TTL_SECONDS: Final[int] = 60 * 60  # 1 hour


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

    async def complete(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object]:
        return await _gemini_call_interactive(self._client, self.model, messages)

    async def complete_background(
        self, messages: list[dict[str, str]]
    ) -> tuple[str, object]:
        return await _gemini_call_background(self._client, self.model, messages)

    async def ping(self) -> None:
        """Lightweight connectivity check."""
        await self._client.aio.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            extra_body={"max_completion_tokens": 1},
        )


# ---------------------------------------------------------------------------
# Provider-agnostic layer — session management, rolling summary, tracing
# ---------------------------------------------------------------------------

class LLMProvider:
    def __init__(self) -> None:
        self._backend = _GeminiClient()
        self.model = self._backend.model

        # session_id -> (summary, last_access_time)
        self.sessions: dict[str, tuple[str, float]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def _evict_stale_sessions(self) -> None:
        cutoff = time.monotonic() - _SESSION_TTL_SECONDS
        stale = [sid for sid, (_, ts) in self.sessions.items() if ts < cutoff]
        for sid in stale:
            self.sessions.pop(sid, None)
            self._session_locks.pop(sid, None)

    def _get_summary(self, session_id: str) -> str:
        entry = self.sessions.get(session_id)
        if entry is None:
            return ""
        summary, _ = entry
        self.sessions[session_id] = (summary, time.monotonic())
        return summary

    def _set_summary(self, session_id: str, summary: str) -> None:
        self.sessions[session_id] = (summary, time.monotonic())
        self._evict_stale_sessions()

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
        user_message: str,
        assistant_reply: str,
    ) -> str:
        user_content = (
            f"Previous summary:\n{current_summary.strip() or '(none yet)'}\n\n"
            f"New user message:\n{user_message}\n\n"
            f"Assistant reply:\n{assistant_reply}\n\n"
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
        summary_used_for_turn: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        langfuse = get_client()
        with langfuse.start_as_current_observation(
            as_type="span",
            name="rolling-summary-update",
        ) as span:
            span.update(
                input={
                    "session_id": session_id,
                    "user_message": user_message,
                    "assistant_reply_preview": assistant_reply[:2000],
                }
            )
            t0 = time.perf_counter()
            try:
                updated = await self.update_rolling_summary(
                    summary_used_for_turn, user_message, assistant_reply
                )
            except Exception:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                span.update(
                    metadata={
                        "summary_refresh_latency_ms": elapsed_ms,
                        "summary_refresh_status": "error",
                    }
                )
                log_rolling_summary_refresh_failed(session_id, elapsed_ms)
                return

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            span.update(
                metadata={
                    "summary_refresh_latency_ms": elapsed_ms,
                    "summary_refresh_status": "ok",
                }
            )
            log_rolling_summary_refresh_complete(session_id, elapsed_ms)

            text = updated.strip()
            if not text:
                return
            async with self._locks_guard:
                lock = self._session_locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                self._set_summary(session_id, text)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        langfuse = get_client()
        with langfuse.start_as_current_observation(as_type="span", name="chat-turn"):
            with propagate_attributes(session_id=request.session_id):
                async with self._locks_guard:
                    lock = self._session_locks.setdefault(request.session_id, asyncio.Lock())
                async with lock:
                    summary = self._get_summary(request.session_id)

                user_turn = (
                    f"Conversation summary so far:\n{summary.strip()}\n\n"
                    f"User message:\n{request.message}"
                    if summary.strip()
                    else request.message
                )

                langfuse.update_current_span(
                    input={
                        "user_message": request.message,
                        "had_prior_summary": bool(summary.strip()),
                    }
                )

                reply = await self._chat_completion_text(
                    [
                        {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_turn},
                    ],
                    observation_name="llm-chat-reply",
                )

                langfuse.update_current_span(output={"reply": reply})

                parent_ctx = otel_context_current()
                asyncio.create_task(
                    self._run_in_context(
                        parent_ctx,
                        self._background_refresh_summary(
                            request.session_id, summary, request.message, reply
                        ),
                    )
                )
                return ChatResponse(reply=reply)

    @staticmethod
    async def _run_in_context(ctx, coro) -> None:
        with otel_attached(ctx):
            await coro

    async def check_readiness(self) -> None:
        await self._backend.ping()
