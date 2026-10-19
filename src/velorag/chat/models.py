from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import relationship

from velorag.db.base import Base


class ChatSessionModel(Base):
    __tablename__ = "velorag_chat_sessions"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    kb_id = Column(String, ForeignKey("velorag_knowledge_bases.id"), nullable=False, index=True)
    name = Column(String(128), default="New Conversation")
    system_prompt = Column(Text, default="You are a helpful grounded assistant.")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages = relationship(
        "ChatMessageModel", back_populates="session", cascade="all, delete-orphan"
    )


class ChatMessageModel(Base):
    __tablename__ = "velorag_chat_messages"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    session_id = Column(String, ForeignKey("velorag_chat_sessions.id"), nullable=False, index=True)
    role = Column(String(32), nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    sources = Column(JSON, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSessionModel", back_populates="messages")
