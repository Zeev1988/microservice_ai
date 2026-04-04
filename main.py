from contextlib import asynccontextmanager
import asyncio
import hashlib
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from google.genai.errors import ClientError

from logging_setup import RequestLoggingMiddleware, configure_logging, log_rate_limit_exceeded
from llm_provider import LLMProvider
from schemas import ChatRequest, ChatResponse
from session_store import SessionStore
from tracing import get_client

load_dotenv()
configure_logging()

# ---------------------------------------------------------------------------
# Security: API key auth
# ---------------------------------------------------------------------------

_API_KEY = os.getenv("API_KEY", "")


def _credential_fingerprint(raw_key: str) -> str:
    """One-way id for rate limits / logs — never put the raw API key in Redis or log fields."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate X-API-Key header. Returns a SHA-256 hex digest (not the secret) for rate limiting."""
    if not _API_KEY:
        raise RuntimeError("API_KEY env var is not set")
    if x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return _credential_fingerprint(x_api_key)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = SessionStore.from_env()
    app.state.provider = LLMProvider(store)
    yield
    await store.close()
    get_client().shutdown()


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.provider


# ---------------------------------------------------------------------------
# Security: rate limiting for destructive actions
# (defined after get_provider so Depends(get_provider) resolves at import time)
# ---------------------------------------------------------------------------

# Allow at most 1 reset per 5 minutes per API key.
_RESET_RATE_LIMIT = 1
_RESET_RATE_WINDOW_SECONDS = 60 * 5


async def check_reset_rate_limit(
    provider: LLMProvider = Depends(get_provider),
    credential_fingerprint: str = Depends(verify_api_key),
) -> None:
    limited = await provider._store.is_rate_limited(
        action="reset",
        client_id=credential_fingerprint,
        limit=_RESET_RATE_LIMIT,
        window_seconds=_RESET_RATE_WINDOW_SECONDS,
    )
    if limited:
        log_rate_limit_exceeded("reset", credential_fingerprint)
        raise HTTPException(status_code=429, detail="Too many reset requests. Try again later.")


# ---------------------------------------------------------------------------
# App and routes
# ---------------------------------------------------------------------------

app = FastAPI(title="AI microservice", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(provider: LLMProvider = Depends(get_provider)) -> dict[str, str]:
    try:
        await asyncio.wait_for(provider.check_readiness(), timeout=5)
    except (httpx.HTTPError, ClientError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Service not ready") from exc
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, provider: LLMProvider = Depends(get_provider)
) -> ChatResponse:
    return await provider.chat(body)


@app.delete(
    "/sessions/{session_id}",
    status_code=204,
    dependencies=[Depends(verify_api_key), Depends(check_reset_rate_limit)],
    summary="GDPR forget — soft-deletes session state (archived 24 h before final expiry)",
)
async def reset_session(
    session_id: str,
    provider: LLMProvider = Depends(get_provider),
) -> None:
    await provider.reset_session(session_id)
