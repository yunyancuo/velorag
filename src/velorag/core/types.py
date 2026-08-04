from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class ChunkStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"


class EmbeddingBackend(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    LANCE = "lance"
    FAISS = "faiss"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"


class RerankerType(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    COHERE = "cohere"
    LLM_BASED = "llm_based"
    RECIPROCAL = "reciprocal_rank_fusion"


@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    retrieval_source: str = ""


@dataclass
class QueryContext:
    query: str
    top_k: int = 10
    mode: RetrievalMode = RetrievalMode.HYBRID
    filters: dict[str, Any] = field(default_factory=dict)
    reranker: Optional[RerankerType] = None
    use_graph: bool = True


@dataclass
class EntityNode:
    id: str
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)


@dataclass
class RelationEdge:
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class GenerationRequest:
    query: str
    context_chunks: list[Chunk]
    system_prompt: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False


@dataclass
class GenerationResponse:
    content: str
    sources: list[SearchResult] = field(default_factory=list)
    token_count: int = 0
