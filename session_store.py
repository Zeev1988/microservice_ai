"""Redis-backed session store: data structures, serialisation, distributed locking, and rate limiting."""

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

# Soft-deleted sessions are kept for 24 h before Redis expires them.
_PENDING_DELETE_TTL_SECONDS: Final[int] = 60 * 60 * 24


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

    @staticmethod
    def _pending_delete_key(session_id: str) -> str:
        return f"session:pending_delete:{session_id}"

    @staticmethod
    def _rate_limit_key(action: str, client_id: str) -> str:
        return f"rate_limit:{action}:{client_id}"

    @staticmethod
    def _research_notes_key(session_id: str) -> str:
        return f"session:notes:{session_id}"

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

    async def save_research_note(self, session_id: str, note: str) -> None:
        """Append a research note to this session in Redis."""
        key = self._research_notes_key(session_id)
        await self._redis.rpush(key, note)
        await self._redis.expire(key, SESSION_TTL_SECONDS)

    async def soft_delete_state(self, session_id: str) -> bool:
        """Soft-delete: archive live state for 24 h, then remove the live key.

        Returns True if a live session existed, False if there was nothing to delete.
        The archived copy lets ops / support recover accidental deletions within the window.
        """
        async with self.session_lock(session_id):
            raw = await self._redis.get(self._key(session_id))
            if raw is None:
                return False
            await self._redis.set(
                self._pending_delete_key(session_id),
                raw,
                ex=_PENDING_DELETE_TTL_SECONDS,
            )
            await self._redis.delete(self._key(session_id))
        return True

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def is_rate_limited(
        self, action: str, client_id: str, *, limit: int, window_seconds: int
    ) -> bool:
        """Fixed-window rate check. Returns True if the caller has exceeded the limit.

        ``client_id`` should be a non-secret stable id (e.g. SHA-256 of API key), never the raw key.
        Uses Redis INCR + EXPIRE so the counter is distributed across all workers.
        """
        key = self._rate_limit_key(action, client_id)
        count = await self._redis.incr(key)
        if count == 1:
            # First hit — set the expiry for this window.
            await self._redis.expire(key, window_seconds)
        return count > limit

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
