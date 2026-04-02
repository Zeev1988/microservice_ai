"""Logging setup, HTTP access middleware, and LLM log event helpers."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

log = logging.getLogger("http.access")
_llm_log = logging.getLogger("llm")


def log_llm_chat_completion(
    observation_name: str,
    usage: dict[str, int] | None,
) -> None:
    extra: dict[str, Any] = {
        "observation": observation_name,
        "total_tokens": (usage or {}).get("total_tokens"),
    }
    if usage:
        for key in ("input_tokens", "output_tokens"):
            if key in usage:
                extra[key] = usage[key]
    _llm_log.info("llm_chat_completion", extra=extra)


def log_rolling_summary_refresh_complete(session_id: str, latency_ms: float) -> None:
    _llm_log.info(
        "rolling_summary_refresh_complete",
        extra={"session_id": session_id, "summary_refresh_latency_ms": latency_ms},
    )


def log_rolling_summary_refresh_failed(session_id: str, latency_ms: float) -> None:
    """Call from inside an ``except`` block so the traceback is recorded."""
    _llm_log.exception(
        "rolling_summary_refresh_failed",
        extra={"session_id": session_id, "summary_refresh_latency_ms": latency_ms},
    )


class JsonLogFormatter(logging.Formatter):
    """One JSON object per line (Loki / ELK / CloudWatch friendly)."""

    RESERVED = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self.RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    root.setLevel(level)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())
    root.addHandler(handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        return response
