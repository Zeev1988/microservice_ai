from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    """Structured input: instruction and document content are separate fields."""

    instruction: str = Field(
        ...,
        min_length=1,
        description="What to do with the document (e.g. summarize for an executive audience).",
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The full document text to process.",
    )


class SummarizeResponse(BaseModel):
    tldr: str = Field(..., min_length=1, description="A TL;DR summary of the document.")
    bullets: list[str] = Field(..., description="A list of bullet points summarizing the document.")


class ChatRequest(BaseModel):
    """Conversation turn with session-scoped history."""

    session_id: str = Field(..., min_length=1, description="Client-owned session identifier.")
    message: str = Field(..., min_length=1, description="User message for this turn.")


class ChatResponse(BaseModel):
    reply: str = Field(..., min_length=1, description="Assistant reply for this turn.")