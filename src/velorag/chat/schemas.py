from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    kb_id: str = Field(default="default", min_length=1, max_length=64)
    name: str = Field(default="New Conversation", min_length=1, max_length=128)
    system_prompt: Optional[str] = Field(
        default="You are a helpful grounded assistant.", max_length=4000
    )


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kb_id: str
    owner_id: Optional[str] = None
    name: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    stream: bool = False
    mode: Literal["dense", "sparse", "hybrid", "graph_enhanced"] = "graph_enhanced"
    top_k: int = Field(default=5, ge=1, le=50)
