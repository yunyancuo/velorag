from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

from velorag.core.types import (
    ChunkingStrategy,
    EmbeddingBackend,
    RerankerType,
    VectorStoreBackend,
)


@dataclass
class ChunkingConfig:
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
    )


@dataclass
class EmbeddingConfig:
    backend: EmbeddingBackend = EmbeddingBackend.SENTENCE_TRANSFORMER
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class VectorStoreConfig:
    backend: VectorStoreBackend = VectorStoreBackend.IN_MEMORY
    collection_name: str = "velorag_default"
    persist_directory: Optional[str] = None


@dataclass
class RetrievalConfig:
    top_k: int = 10
    dense_weight: float = 0.5
    sparse_weight: float = 0.3
    graph_weight: float = 0.2
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class RerankerConfig:
    enabled: bool = False
    model_type: RerankerType = RerankerType.LEXICAL_OVERLAP
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


@dataclass
class KnowledgeGraphConfig:
    enabled: bool = True
    max_hops: int = 2
    min_entity_Length: int = 2
    co_occurrence_window: int = 3


@dataclass
class GenerationConfig:
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    max_tokens: int = 1024
    temperature: float = 0.3
    system_prompt: str = (
        "You are a precise research assistant. Answer the question based ONLY on "
        "the provided context. Cite your sources using [Source N] notation. "
        "If the context is insufficient, state clearly that you cannot answer."
    )


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class VeloRAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    knowledge_graph: KnowledgeGraphConfig = field(default_factory=KnowledgeGraphConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VeloRAGConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, data: dict) -> VeloRAGConfig:
        cfg = cls()
        if "chunking" in data:
            c = data["chunking"]
            cfg.chunking = ChunkingConfig(
                strategy=ChunkingStrategy(c.get("strategy", "recursive")),
                chunk_size=c.get("chunk_size", 512),
                chunk_overlap=c.get("chunk_overlap", 64),
            )
        if "embedding" in data:
            e = data["embedding"]
            cfg.embedding = EmbeddingConfig(
                backend=EmbeddingBackend(e.get("backend", "sentence_transformer")),
                model_name=e.get("model_name", "all-MiniLM-L6-v2"),
                dimension=e.get("dimension", 384),
                batch_size=e.get("batch_size", 32),
                api_key=e.get("api_key"),
                api_base=e.get("api_base"),
            )
        if "vector_store" in data:
            v = data["vector_store"]
            cfg.vector_store = VectorStoreConfig(
                backend=VectorStoreBackend(v.get("backend", "in_memory")),
                collection_name=v.get("collection_name", "velorag_default"),
                persist_directory=v.get("persist_directory"),
            )
        if "retrieval" in data:
            r = data["retrieval"]
            cfg.retrieval = RetrievalConfig(**{k: v for k, v in r.items() if k in RetrievalConfig.__dataclass_fields__})
        if "reranker" in data:
            rk = data["reranker"]
            cfg.reranker = RerankerConfig(
                enabled=rk.get("enabled", False),
                model_type=RerankerType(rk.get("model_type", "lexical_overlap")),
                model_name=rk.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                top_n=rk.get("top_n", 5),
            )
        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            cfg.knowledge_graph = KnowledgeGraphConfig(**{k: v for k, v in kg.items() if k in KnowledgeGraphConfig.__dataclass_fields__})
        if "generation" in data:
            g = data["generation"]
            cfg.generation = GenerationConfig(**{k: v for k, v in g.items() if k in GenerationConfig.__dataclass_fields__})
        if "server" in data:
            s = data["server"]
            cfg.server = ServerConfig(**{k: v for k, v in s.items() if k in ServerConfig.__dataclass_fields__})
        return cfg
