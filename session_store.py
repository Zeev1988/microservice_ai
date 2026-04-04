"""Redis-backed session store: data structures, serialisation, and distributed locking."""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Final

import redis.asyncio as aioredis

# Sessions expire in Redis after this period of inactivity.
SESSION_TTL_SECONDS: Final[int] = 60 * 60  # 1 hour

# Max time any process may hold a session lock; prevents deadlock on crash.
_LOCK_TIMEOUT_SECONDS: Final[float] = 30.0


@dataclass
class Exchange:
    user: str
    assistant: str


@dataclass
class SessionState:
    summary: str = ""
    buffer: list[Exchange] = field(default_factory=list)


class SessionStore:
    """Async Redis wrapper for session state and distributed per-session locking."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    @classmethod
    def from_env(cls) -> "SessionStore":
        url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return cls(aioredis.from_url(url, decode_responses=True))

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def _lock_key(session_id: str) -> str:
        return f"session_lock:{session_id}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def get_state(self, session_id: str) -> SessionState:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return SessionState()
        data = json.loads(raw)
        return SessionState(
            summary=data.get("summary", ""),
            buffer=[Exchange(**ex) for ex in data.get("buffer", [])],
        )

    async def set_state(self, session_id: str, state: SessionState) -> None:
        payload = json.dumps({
            "summary": state.summary,
            "buffer": [{"user": ex.user, "assistant": ex.assistant} for ex in state.buffer],
        })
        await self._redis.set(self._key(session_id), payload, ex=SESSION_TTL_SECONDS)

    # ------------------------------------------------------------------
    # Distributed lock
    # ------------------------------------------------------------------

    @contextlib.asynccontextmanager
    async def session_lock(self, session_id: str):
        """Distributed Redis lock for a session. Safe across multiple workers."""
        lock = self._redis.lock(self._lock_key(session_id), timeout=_LOCK_TIMEOUT_SECONDS)
        async with lock:
            yield

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def ping(self) -> None:
        await self._redis.ping()

    async def close(self) -> None:
        await self._redis.aclose()
