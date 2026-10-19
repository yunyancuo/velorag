from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import uuid


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    FIXED_SIZE = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


ChunkStrategy = ChunkingStrategy


class RetrievalMode(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    GRAPH_ENHANCED = "graph_enhanced"
    LOCAL_RESIDUAL = "local_residual"
    RESIDUAL_HYBRID = "residual_hybrid"


class EmbeddingBackend(str, Enum):
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    HASH = "hash"


class VectorStoreBackend(str, Enum):
    CHROMA = "chroma"
    IN_MEMORY = "in_memory"
    MEMORY = "in_memory"


class RerankerType(str, Enum):
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    LEXICAL_OVERLAP = "lexical_overlap"


@dataclass
class Document:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def __post_init__(self):
        if self.content and not self.text:
            self.text = self.content
        elif self.text and not self.content:
            self.content = self.text


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    text: str = ""
    doc_id: str = ""
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    start_char: int = 0
    end_char: int = 0


@dataclass
class SearchResult:
    chunk: Chunk
    score: float = 0.0
    retrieval_source: str = "unknown"


@dataclass
class QueryContext:
    query: str
    top_k: int = 5
    mode: RetrievalMode | str = RetrievalMode.HYBRID
    use_graph: bool = True
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    """Verified inline citation linking generated claims back to a retrieved chunk."""
    index: int
    chunk_id: str
    doc_id: str
    source: str
    quote: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityNode:
    id: str
    name: str
    entity_type: str = "concept"
    chunk_ids: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationEdge:
    source_id: str
    target_id: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    relation: str = "related_to"
    relation_type: str = "related_to"
    weight: float = 1.0
    chunk_id: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRequest:
    query: str
    context_chunks: list[Chunk] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.3
    system_prompt: Optional[str] = None
    min_relevance_score: float = 0.12
    strict_citation: bool = True
    stream: bool = False


@dataclass
class GenerationResponse:
    content: str
    sources: list[SearchResult] = field(default_factory=list)
    model: str = ""
    token_count: int = 0
    citations: list[Citation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: Optional[str] = None
    confidence: float = 1.0
