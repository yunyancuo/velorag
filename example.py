"""VeloRAG 快速上手示例：本地文档入库、三路混合检索与可验证引用问答。"""

from pathlib import Path

from velorag import VeloRAG
from velorag.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    KnowledgeGraphConfig,
    RerankerConfig,
    VectorStoreConfig,
    VeloRAGConfig,
)
from velorag.core.types import (
    ChunkingStrategy,
    EmbeddingBackend,
    RerankerType,
    VectorStoreBackend,
)


def main() -> None:
    cfg = VeloRAGConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.RECURSIVE, chunk_size=300, chunk_overlap=40
        ),
        embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=256),
        vector_store=VectorStoreConfig(backend=VectorStoreBackend.IN_MEMORY),
        reranker=RerankerConfig(enabled=True, model_type=RerankerType.LEXICAL_OVERLAP),
        knowledge_graph=KnowledgeGraphConfig(enabled=True, max_hops=2),
    )
    rag = VeloRAG(config=cfg)

    readme = Path("README.md")
    if readme.exists():
        rag.ingest_file_sync(str(readme), metadata={"kb_id": "demo"})
    else:
        rag.ingest_text(
            text="VeloRAG combines Dense vector search, multilingual BM25, and Knowledge Graph RRF fusion.",
            doc_id="demo_doc",
            source="demo.md",
            metadata={"kb_id": "demo"},
        )

    results = rag.search(
        "What is VeloRAG?",
        top_k=5,
        mode="graph_enhanced",
        filters={"kb_id": "demo"},
    )
    for r in results:
        print(f"[{r.score:.3f}] ({r.retrieval_source}) {r.chunk.text[:100]}...")

    answer = rag.query(
        "What retrieval modes does VeloRAG combine?",
        top_k=3,
        mode="graph_enhanced",
        filters={"kb_id": "demo"},
    )
    print(f"\nAnswer (refused={answer.refused}, confidence={answer.confidence:.2f}):")
    print(answer.content)


if __name__ == "__main__":
    main()
