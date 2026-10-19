from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from velorag.db.base import Base


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class KnowledgeBaseModel(Base):
    __tablename__ = "velorag_knowledge_bases"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    name = Column(String(128), nullable=False, index=True)
    description = Column(Text, default="")
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, default=64)
    embedding_model = Column(String(128), default="all-MiniLM-L6-v2")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    documents = relationship(
        "DocumentModel", back_populates="kb", cascade="all, delete-orphan"
    )


class DocumentModel(Base):
    __tablename__ = "velorag_documents"

    id = Column(String, primary_key=True, default=lambda: uuid.uuid4().hex[:12])
    kb_id = Column(String, ForeignKey("velorag_knowledge_bases.id"), nullable=False, index=True)
    filename = Column(String(256), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    error_msg = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    kb = relationship("KnowledgeBaseModel", back_populates="documents")
