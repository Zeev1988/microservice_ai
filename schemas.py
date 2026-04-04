from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Conversation turn with session-scoped history."""

    session_id: str = Field(..., min_length=1, description="Client-owned session identifier.")
    message: str = Field(..., min_length=1, description="User message for this turn.")


class ChatResponse(BaseModel):
    session_id: str = Field(..., min_length=1, description="Client-owned session identifier.")
    reply: str = Field(..., min_length=1, description="Assistant reply for this turn.")
