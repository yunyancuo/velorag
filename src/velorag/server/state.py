from __future__ import annotations

from typing import Optional
from velorag.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    KnowledgeGraphConfig,
    RerankerConfig,
    VeloRAGConfig,
    VectorStoreConfig,
)
from velorag.core.types import (
    ChunkingStrategy,
    EmbeddingBackend,
    RerankerType,
    VectorStoreBackend,
)
from velorag.pipeline import VeloRAG

_DEFAULT_PIPELINE: Optional[VeloRAG] = None


def get_default_pipeline(seed_builtin: bool = True) -> VeloRAG:
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        cfg = VeloRAGConfig(
            chunking=ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=300, chunk_overlap=40),
            embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=256),
            vector_store=VectorStoreConfig(backend=VectorStoreBackend.IN_MEMORY),
            reranker=RerankerConfig(enabled=True, model_type=RerankerType.LEXICAL_OVERLAP),
            knowledge_graph=KnowledgeGraphConfig(enabled=True, max_hops=2),
            generation=GenerationConfig(model_name="gpt-4o-mini"),
        )
        _DEFAULT_PIPELINE = VeloRAG(config=cfg)
        if seed_builtin:
            from velorag.eval.dataset import get_builtin_campus_benchmark
            docs, _ = get_builtin_campus_benchmark()
            for doc in docs:
                _DEFAULT_PIPELINE.ingest_text(
                    text=doc.text,
                    doc_id=doc.doc_id,
                    source=doc.metadata.get("source", doc.title),
                    metadata={**doc.metadata, "title": doc.title, "kb_id": "default"},
                )
    return _DEFAULT_PIPELINE


def reset_default_pipeline(seed_builtin: bool = False) -> VeloRAG:
    global _DEFAULT_PIPELINE
    _DEFAULT_PIPELINE = None
    return get_default_pipeline(seed_builtin=seed_builtin)
