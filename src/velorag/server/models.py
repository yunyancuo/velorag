from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    mode: str = "hybrid"
    use_graph: bool = True
    filters: dict = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    results: list[ResultItem]
    total: int
    elapsed_ms: float


class ResultItem(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = "hybrid"
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class GenerateResponse(BaseModel):
    query: str
    answer: str
    sources: list[ResultItem]
    token_count: int


class IndexStatus(BaseModel):
    total_chunks: int
    total_entities: int
    total_relations: int
    shard_count: int
