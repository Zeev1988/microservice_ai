"""ARQ worker — background task definitions and startup/shutdown hooks.

Run with:
    arq worker.WorkerSettings
"""

from __future__ import annotations

import os
from typing import Any

from arq.connections import RedisSettings

from session_store import Exchange, SessionStore
from vector_store import VectorStore
from llm_provider import LLMProvider

async def refresh_summary(
    ctx: dict[str, Any],
    session_id: str,
    snapshot_summary: str,
    snapshot_buffer: list[dict[str, str]],
) -> None:
    """Recompute and persist the rolling conversation summary for *session_id*.

    Runs in the ARQ worker process. Deserialises the exchange dicts back into
    Exchange objects before delegating to the provider's existing logic.
    """
    provider: LLMProvider = ctx["provider"]
    buffer = [Exchange(**ex) for ex in snapshot_buffer]
    await provider._background_refresh_summary(session_id, snapshot_summary, buffer)


async def on_startup(ctx: dict[str, Any]) -> None:
    store = SessionStore.from_env()
    vector_store = VectorStore.from_env()
    ctx["provider"] = LLMProvider(store, vector_store)


async def on_shutdown(ctx: dict[str, Any]) -> None:
    provider = ctx.get("provider")
    if provider is not None:
        await provider._store.close()


class WorkerSettings:
    functions = [refresh_summary]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379")
    )
