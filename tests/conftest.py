"""Shared fixtures: fake Redis, stubbed LLMProvider, and an httpx.AsyncClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import fakeredis.aioredis as fakeredis
import pytest
from httpx import ASGITransport, AsyncClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Set env vars before importing the app so FastAPI / main.py reads them.
os.environ.setdefault("API_KEY", "test-secret")
os.environ.setdefault("GEMINI_API_KEY", "fake")
os.environ.setdefault("GEMINI_MODEL", "gemini-pro")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "fake-pub")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "fake-sec")

from main import app
from session_store import SessionStore


@pytest.fixture
def fake_redis():
    """In-process fake Redis — no real Redis required."""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def session_store(fake_redis):
    return SessionStore(fake_redis)


@pytest.fixture
def stub_llm_provider(session_store):
    """LLMProvider with the Gemini backend replaced by a simple stub."""
    from llm_provider import LLMProvider

    provider = LLMProvider.__new__(LLMProvider)
    provider._store = session_store
    provider.model = "stub-model"
    # Replace the real LLM call with a simple echo stub.
    provider._chat_completion_text = AsyncMock(return_value="Hello from stub!")
    return provider


@pytest.fixture
async def client(stub_llm_provider):
    """AsyncClient wired to the FastAPI app, with all external deps stubbed."""
    app.state.provider = stub_llm_provider
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
