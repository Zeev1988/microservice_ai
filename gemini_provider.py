import asyncio
import os
from typing import Final

import httpx
from google import genai
from google.genai.errors import ClientError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from schemas import ChatRequest, ChatResponse

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


class InvalidGeminiResponseError(ValueError):
    pass


def _is_invalid_response_error(exc: BaseException) -> bool:
    return isinstance(exc, InvalidGeminiResponseError)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if _is_invalid_response_error(exc):
        return True
    return isinstance(exc, ClientError) and exc.code == 429


class GeminiProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL")

        self.client = genai.Client(api_key=self.api_key)
        self.sessions: dict[str, str] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _chat_completion_text(self, messages: list[dict[str, str]]) -> str:
        response = await self.client.aio.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            raise InvalidGeminiResponseError("Gemini returned empty chat content")
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
            ]
        )

    async def _background_refresh_summary(
        self,
        session_id: str,
        summary_used_for_turn: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        try:
            updated = await self.update_rolling_summary(
                summary_used_for_turn, user_message, assistant_reply
            )
            text = updated.strip()
            if not text:
                return
            async with self._locks_guard:
                lock = self._session_locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                self.sessions[session_id] = text
        except Exception:
            pass

    async def chat(self, request: ChatRequest) -> ChatResponse:
        async with self._locks_guard:
            lock = self._session_locks.setdefault(request.session_id, asyncio.Lock())
        async with lock:
            summary = self.sessions.get(request.session_id, "")

        if summary.strip():
            user_turn = (
                f"Conversation summary so far:\n{summary.strip()}\n\n"
                f"User message:\n{request.message}"
            )
        else:
            user_turn = request.message

        reply = await self._chat_completion_text(
            [
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_turn},
            ]
        )

        asyncio.create_task(
            self._background_refresh_summary(
                request.session_id, summary, request.message, reply
            )
        )
        return ChatResponse(reply=reply)

    async def check_readiness(self) -> None:
        # Lightweight dependency check for auth/connectivity to Gemini.
        await self.client.aio.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            extra_body={"max_completion_tokens": 1},
        )
