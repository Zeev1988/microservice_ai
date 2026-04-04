"""API tests: integration, security, and rate limiting.

Strategy
--------
* LLM calls are stubbed — we are testing the *plumbing* (auth, routing, Redis),
  not the quality of Gemini's output.
* Redis is replaced by fakeredis, so tests are hermetic and fast.
"""

VALID_KEY = "test-secret"
HEADERS = {"X-API-Key": VALID_KEY}


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

async def test_chat_returns_200_with_valid_response(client):
    """A valid /chat request returns 200 and a well-formed ChatResponse."""
    response = await client.post(
        "/chat",
        json={"session_id": "session-abc", "message": "Hello!"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "session-abc"
    assert isinstance(body["reply"], str)
    assert len(body["reply"]) > 0


# ---------------------------------------------------------------------------
# Security test
# ---------------------------------------------------------------------------

async def test_reset_without_api_key_returns_422(client):
    """DELETE /sessions without X-API-Key header returns 422: the header is required.

    FastAPI validates the declared Header(...) before auth logic runs, so a
    completely missing required header is a request-validation error (422), not
    an auth error (401). The correct way to trigger a 401 is to send a *wrong* key.
    """
    response = await client.delete("/sessions/session-abc")
    assert response.status_code == 422


async def test_reset_with_wrong_api_key_returns_401(client):
    """Wrong API key on a protected endpoint returns 401."""
    response = await client.delete(
        "/sessions/session-abc",
        headers={"X-API-Key": "wrong-key"},
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Rate limit test
# ---------------------------------------------------------------------------

async def test_reset_rate_limit_returns_429_on_second_call(client):
    """Second reset within the rate-limit window returns 429 Too Many Requests."""
    first = await client.delete("/sessions/session-abc", headers=HEADERS)
    # First call should succeed (204 No Content).
    assert first.status_code == 204

    second = await client.delete("/sessions/session-abc", headers=HEADERS)
    # Second call within the window is blocked.
    assert second.status_code == 429
    assert "Too many" in second.json()["detail"]
