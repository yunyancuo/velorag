from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON, UniqueConstraint, Index,
)
from sqlalchemy.orm import relationship

from velorag.db.base import Base


def _uuid():
    return uuid.uuid4().hex


def _now():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(String(32), primary_key=True, default=_uuid)
    username = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(256), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", back_populates="owner", cascade="all, delete-orphan")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(32), primary_key=True, default=_uuid)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    key_hash = Column(String(256), unique=True, nullable=False)
    key_prefix = Column(String(8), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="api_keys")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(String(32), primary_key=True, default=_uuid)
    name = Column(String(256), nullable=False)
    description = Column(Text, default="")
    owner_id = Column(String(32), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding_model = Column(String(128), default="all-MiniLM-L6-v2")
    embedding_dim = Column(Integer, default=384)
    chunk_strategy = Column(String(32), default="recursive")
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, default=64)
    retrieval_mode = Column(String(32), default="hybrid")
    graph_enabled = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    owner = relationship("User", back_populates="knowledge_bases")
    documents = relationship("Document", back_populates="knowledge_base", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_kb_owner_name"),
    )


class Document(Base):
    __tablename__ = "documents"

    id = Column(String(32), primary_key=True, default=_uuid)
    kb_id = Column(String(32), ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(16), nullable=False)
    file_size = Column(Integer, default=0)
    file_hash = Column(String(64), nullable=True)
    status = Column(String(16), default="pending", nullable=False, index=True)
    chunk_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(String(32), primary_key=True, default=_uuid)
    doc_id = Column(String(32), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    kb_id = Column(String(32), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    token_count = Column(Integer, default=0)
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_kb_id_hash", "kb_id"),
    )


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(String(32), primary_key=True, default=_uuid)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    kb_id = Column(String(32), nullable=False, index=True)
    query = Column(Text, nullable=False)
    mode = Column(String(16), default="hybrid")
    top_k = Column(Integer, default=10)
    result_count = Column(Integer, default=0)
    elapsed_ms = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)


class FeedbackLog(Base):
    __tablename__ = "feedback_logs"

    id = Column(String(32), primary_key=True, default=_uuid)
    user_id = Column(String(32), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    kb_id = Column(String(32), nullable=False, index=True)
    query = Column(Text, nullable=False)
    chunk_id = Column(String(32), nullable=True)
    rating = Column(Integer, default=0)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
