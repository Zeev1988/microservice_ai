from contextlib import asynccontextmanager
import asyncio

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from google.genai.errors import ClientError
import httpx

load_dotenv()

from logging_setup import RequestLoggingMiddleware, configure_logging

configure_logging()

from llm_provider import LLMProvider
from tracing import get_client
from schemas import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = LLMProvider()
    yield
    get_client().shutdown()


def get_provider(request: Request) -> LLMProvider:
    return request.app.state.provider


app = FastAPI(title="AI microservice", version="0.1.0", lifespan=lifespan)
app.add_middleware(RequestLoggingMiddleware)


@app.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready(provider: GeminiProvider = Depends(get_provider)) -> dict[str, str]:
    try:
        await asyncio.wait_for(provider.check_readiness(), timeout=5)
    except (httpx.HTTPError, ClientError, asyncio.TimeoutError) as exc:
        raise HTTPException(status_code=503, detail="Gemini provider not ready") from exc
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, provider: GeminiProvider = Depends(get_provider)
) -> ChatResponse:
    return await provider.chat(body)
