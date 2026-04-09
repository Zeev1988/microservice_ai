"""Shared fixtures: fake Redis, stubbed LLMProvider, and an httpx.AsyncClient."""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from unittest.mock import AsyncMock

import fakeredis.aioredis as fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force-set all values tests depend on so CI env vars cannot override them.
os.environ["API_KEY"] = "test-secret"
os.environ.setdefault("GEMINI_API_KEY", "fake")
os.environ.setdefault("GEMINI_MODEL", "gemini-pro")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "fake-pub")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "fake-sec")
# Prevent Langfuse / OTLP from exporting during tests (fake creds → 401 and
# exporter errors break ASGI handling under Python 3.11+ exception groups).
os.environ["OTEL_SDK_DISABLED"] = "true"

from main import app  # noqa: E402
from session_store import SessionStore  # noqa: E402


@pytest.fixture
def fake_redis():
    """In-process fake Redis — no real Redis required."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def session_store(fake_redis):
    """SessionStore backed by fakeredis, with the Lua-script lock replaced.

    redis-py's Lock uses EVALSHA (Lua scripting) internally.  fakeredis does
    not reliably support it, so we swap in a plain asyncio.Lock for tests.
    """
    store = SessionStore(fake_redis)

    _locks: dict[str, asyncio.Lock] = {}

    @contextlib.asynccontextmanager
    async def _asyncio_lock(session_id: str):
        lock = _locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            yield

    store.session_lock = _asyncio_lock
    return store


@pytest.fixture
def stub_llm_provider(session_store):
    """LLMProvider with the Gemini backend replaced by a simple stub."""
    from llm_provider import LLMProvider

    provider = LLMProvider.__new__(LLMProvider)
    provider._store = session_store
    provider.model = "stub-model"
    provider._chat_completion_text = AsyncMock(return_value="Hello from stub!")
    provider._chat_with_tools = AsyncMock(return_value="Hello from stub!")
    return provider


@pytest.fixture
async def client(stub_llm_provider):
    """AsyncClient wired to the FastAPI app, with all external deps stubbed."""
    app.state.provider = stub_llm_provider
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
