from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from velorag.knowledge.models import DocumentStatus

ChunkStrategyLiteral = Literal["fixed", "sentence", "recursive", "semantic"]
RetrievalModeLiteral = Literal["dense", "sparse", "hybrid", "graph_enhanced"]


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=2000)
    chunk_strategy: ChunkStrategyLiteral = "recursive"
    chunk_size: int = Field(default=512, ge=64, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=4096)
    retrieval_mode: RetrievalModeLiteral = "hybrid"
    embedding_model: str = Field(default="all-MiniLM-L6-v2", min_length=1, max_length=128)
    is_public: Optional[bool] = None

    @model_validator(mode="after")
    def _validate_overlap(self) -> KnowledgeBaseCreate:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class KnowledgeBaseUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    description: Optional[str] = Field(default=None, max_length=2000)
    chunk_strategy: Optional[ChunkStrategyLiteral] = None
    chunk_size: Optional[int] = Field(default=None, ge=64, le=8192)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=4096)
    retrieval_mode: Optional[RetrievalModeLiteral] = None
    is_public: Optional[bool] = None


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    description: str
    owner_id: Optional[str] = None
    is_public: bool = True
    chunk_strategy: str = "recursive"
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str = "hybrid"
    embedding_model: str
    document_count: int = 0
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
