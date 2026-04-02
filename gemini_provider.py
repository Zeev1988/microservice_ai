import asyncio
import os
import json
from typing import Final

import httpx
from google import genai
from google.genai.errors import ClientError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from schemas import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse

_SUMMARIZE_SYSTEM_PROMPT: Final[str] = (
    "You are a technical document assistant. Follow the instruction precisely "
    "and base your answer only on the document provided."
    "Return the response as a JSON object with keys 'tldr' and 'bullets'."
)

_CHAT_SYSTEM_PROMPT: Final[str] = (
    "You are a helpful assistant. Answer concisely and use prior turns in the "
    "conversation when relevant."
)


class InvalidGeminiResponseError(ValueError):
    pass


def _is_invalid_response_error(exc: BaseException) -> bool:
    return isinstance(exc, (json.JSONDecodeError, InvalidGeminiResponseError))


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    if _is_invalid_response_error(exc):
        return True
    return isinstance(exc, ClientError) and exc.code == 429


def _parse_summary_response(content: str | None) -> SummarizeResponse:
    if content is None:
        raise InvalidGeminiResponseError("Gemini returned empty message content")
    try:
        return SummarizeResponse.model_validate_json(content)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidGeminiResponseError("Gemini returned invalid summary JSON") from exc


class GeminiProvider:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = os.getenv("GEMINI_MODEL")

        self.client = genai.Client(api_key=self.api_key)
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    async def _chat_completion(
        self, messages: list[dict[str, str]]
    ) -> SummarizeResponse:
        response = await self.client.aio.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"response_mime_type": "application/json"},
        )
        return _parse_summary_response(response.choices[0].message.content)

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

    async def generate_response(self, request: SummarizeRequest) -> SummarizeResponse:
        user_message = (
            f"Instruction:\n{request.instruction}\n\nDocument:\n{request.content}"
        )
        return await self._chat_completion(
            [
                {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        async with self._locks_guard:
            lock = self._session_locks.setdefault(request.session_id, asyncio.Lock())
        async with lock:
            history = self._sessions.setdefault(request.session_id, [])
            history.append({"role": "user", "content": request.message})
            messages: list[dict[str, str]] = [
                {"role": "system", "content": _CHAT_SYSTEM_PROMPT},
                *history,
            ]
            reply = await self._chat_completion_text(messages)
            history.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply)

    async def check_readiness(self) -> None:
        # Lightweight dependency check for auth/connectivity to Gemini.
        await self.client.aio.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "ping"}],
            extra_body={"max_completion_tokens": 1},
        )
