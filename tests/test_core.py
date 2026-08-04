"""Tests for velorag core functionality."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_document_parsing():
    from velorag.document.parser import TextParser, MarkdownParser
    from velorag.document.chunker import Chunker

    parser = TextParser()
    doc = parser.parse(b"Hello world. This is a test.", {"source": "test"})
    assert doc.content == "Hello world. This is a test."
    assert doc.metadata["source"] == "test"

    chunker = Chunker()
    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert chunks[0].text == "Hello world. This is a test."


def test_chunker_strategies():
    from velorag.document.chunker import Chunker
    from velorag.document.parser import TextParser
    from velorag.core.settings import ChunkSettings
    from velorag.core.types import ChunkStrategy, Document

    doc = Document(id="test", content="Sentence one. Sentence two. Sentence three.", source="test")

    # Fixed size
    settings = ChunkSettings(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=20, chunk_overlap=5)
    chunker = Chunker(settings)
    chunks = chunker.chunk(doc)
    assert len(chunks) >= 2

    # Sentence
    settings = ChunkSettings(strategy=ChunkStrategy.SENTENCE, chunk_size=512)
    chunker = Chunker(settings)
    chunks = chunker.chunk(doc)
    assert len(chunks) == 1


def test_embedding_in_memory():
    import pytest
    from velorag.core.settings import EmbeddingSettings
    from velorag.core.types import EmbeddingBackend

    try:
        import sentence_transformers
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    from velorag.embedding.embedder import create_embedder
    settings = EmbeddingSettings(backend=EmbeddingBackend.SENTENCE_TRANSFORMER, model_name="all-MiniLM-L6-v2")

    try:
        embedder = create_embedder(settings)
    except OSError:
        pytest.skip("Cannot download model (offline)")


def test_vector_store():
    from velorag.indexing.vector_store import InMemoryVectorStore
    from velorag.core.types import Chunk

    store = InMemoryVectorStore(dimension=128)
    assert store.count() == 0

    chunk = Chunk(id="c1", text="test chunk", doc_id="d1", chunk_index=0)
    emb = [0.1] * 128
    store.add([chunk], [emb])
    assert store.count() == 1

    results = store.search(emb, top_k=5)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    assert results[0].score > 0.9


def test_bm25_scorer():
    from velorag.retrieval.engine import BM25Scorer
    from velorag.core.types import Chunk

    chunks = [
        Chunk(id="c1", text="The quick brown fox jumps over the lazy dog", doc_id="d1", chunk_index=0),
        Chunk(id="c2", text="Python is a great programming language", doc_id="d2", chunk_index=0),
        Chunk(id="c3", text="Machine learning is transforming industries", doc_id="d3", chunk_index=0),
    ]

    bm25 = BM25Scorer()
    bm25.index(chunks)

    results = bm25.search("quick fox", top_k=2)
    assert len(results) > 0
    assert "fox" in results[0].chunk.text


def test_knowledge_graph():
    from velorag.graph.knowledge_graph import KnowledgeGraph
    from velorag.core.types import Chunk

    kg = KnowledgeGraph()
    chunks = [
        Chunk(id="c1", text="John Smith is a software engineer at Google. Python is a programming language.", doc_id="d1", chunk_index=0),
    ]

    kg.extract_from_chunks(chunks)
    assert kg.get_entity_count() > 0

    entities = kg.find_relevant_entities("John Smith")
    assert len(entities) > 0


def test_cache():
    from velorag.utils.cache import MemoryCache

    cache = MemoryCache(max_entries=5)
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.get("key2") is None
    cache.delete("key1")
    assert cache.get("key1") is None


def test_core_types():
    from velorag.core.types import Document, Chunk, SearchResult, QueryContext

    doc = Document(id="d1", content="test", metadata={"key": "val"})
    assert doc.id == "d1"

    chunk = Chunk(id="c1", text="chunk text", doc_id="d1", chunk_index=0)
    result = SearchResult(chunk=chunk, score=0.95, retrieval_source="dense")
    assert result.chunk.id == "c1"

    ctx = QueryContext(query="test query", top_k=5)
    assert ctx.query == "test query"


def test_settings():
    from velorag.core.settings import Settings

    settings = Settings()
    assert settings.chunk.chunk_size == 512
    assert settings.embedding.model_name == "all-MiniLM-L6-v2"


def test_registry():
    from velorag.core.registry import ComponentRegistry

    reg = ComponentRegistry()
    reg.register("parsers", "text", lambda: "text_parser")
    assert reg.get("parsers", "text")() == "text_parser"
    assert "parsers" in reg.list()


def test_pipeline_smoke():
    import pytest

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    from velorag.pipeline import VeloRAG
    from velorag.core.settings import Settings
    from velorag.core.types import Chunk

    settings = Settings()
    settings.index.num_shards = 1
    settings.graph.enabled = False

    rag = VeloRAG(settings)

    # Create a temp file and ingest
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as f:
        f.write("VeloRAG is a high-performance RAG framework. It supports hybrid search and knowledge graphs. "
                "The framework is written in Python and uses vector stores for fast retrieval. "
                "Machine learning and natural language processing are core components. "
                "BERT embeddings provide semantic search capabilities.")
        tmp_path = f.name

    try:
        chunks = rag.ingest_file(tmp_path)
        assert len(chunks) > 0

        results = rag.search("What is VeloRAG?", top_k=3)
        assert len(results) > 0
        assert any("VeloRAG" in r.chunk.text for r in results)
    finally:
        import os
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_document_parsing()
    test_chunker_strategies()
    test_vector_store()
    test_bm25_scorer()
    test_knowledge_graph()
    test_cache()
    test_core_types()
    test_settings()
    test_registry()
    test_pipeline_smoke()
    print("All tests passed!")
