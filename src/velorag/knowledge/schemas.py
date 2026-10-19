from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

from velorag.knowledge.models import DocumentStatus


class KnowledgeBaseCreate(BaseModel):
    name: str
    description: str = ""
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "all-MiniLM-L6-v2"


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    created_at: datetime
    updated_at: datetime


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    kb_id: str
    filename: str
    file_size: int
    chunk_count: int
    status: DocumentStatus
    error_msg: Optional[str] = None
    created_at: datetime
    updated_at: datetime
