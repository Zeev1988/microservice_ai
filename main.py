from contextlib import asynccontextmanager
import asyncio

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from google.genai.errors import ClientError
import httpx

load_dotenv()

from gemini_provider import GeminiProvider
from schemas import ChatRequest, ChatResponse, SummarizeRequest, SummarizeResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.provider = GeminiProvider()
    yield


def get_provider(request: Request) -> GeminiProvider:
    return request.app.state.provider


app = FastAPI(title="AI microservice", version="0.1.0", lifespan=lifespan)


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


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    body: SummarizeRequest, provider: GeminiProvider = Depends(get_provider)
) -> SummarizeResponse:
    return await provider.generate_response(body)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest, provider: GeminiProvider = Depends(get_provider)
) -> ChatResponse:
    return await provider.chat(body)
