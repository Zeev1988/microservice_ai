"""Trace context: Langfuse client helpers, OTel propagation, and usage mapping."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from langfuse import get_client, propagate_attributes
from opentelemetry import context as context_api
from opentelemetry.context import Context

__all__ = [
    "Context",
    "get_client",
    "propagate_attributes",
    "otel_attached",
    "otel_context_current",
    "usage_details_for_langfuse",
]


def otel_context_current() -> Context:
    return context_api.get_current()


@contextmanager
def otel_attached(parent: Context):
    token = context_api.attach(parent)
    try:
        yield
    finally:
        context_api.detach(token)


def usage_details_for_langfuse(response: Any) -> dict[str, int] | None:
    """Map Gemini / OpenAI-compat ``usage`` onto Langfuse ``usage_details`` keys."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details: dict[str, int] = {}

    def pick(*names: str) -> int | None:
        for n in names:
            v = getattr(usage, n, None)
            if v is not None:
                return int(v)
        return None

    inp = pick("prompt_tokens", "input_tokens")
    out = pick("completion_tokens", "output_tokens")
    total = pick("total_tokens")
    if inp is not None:
        details["input_tokens"] = inp
    if out is not None:
        details["output_tokens"] = out
    if total is not None:
        details["total_tokens"] = total
    elif inp is not None and out is not None:
        details["total_tokens"] = inp + out

    return details or None
