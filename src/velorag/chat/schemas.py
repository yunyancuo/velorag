from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    kb_id: str = "default"
    name: str = "New Conversation"
    system_prompt: Optional[str] = "You are a helpful grounded assistant."


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
    name: str
    system_prompt: str
    created_at: datetime
    updated_at: datetime
    messages: list[ChatMessageResponse] = Field(default_factory=list)


class ChatRequest(BaseModel):
    message: str
    stream: bool = False
    mode: str = "graph_enhanced"
    top_k: int = 5
