from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Optional

from velorag.core.settings import Settings
from velorag.core.types import (
    Chunk,
    Document,
    GenerationRequest,
    GenerationResponse,
    QueryContext,
    RetrievalMode,
    SearchResult,
)
from velorag.document.chunker import Chunker
from velorag.document.loader import DocumentLoader
from velorag.embedding.embedder import BaseEmbedder, create_embedder
from velorag.generation.llm import LLMGenerator
from velorag.graph.knowledge_graph import KnowledgeGraph
from velorag.indexing.sharding import ShardedIndex
from velorag.indexing.vector_store import BaseVectorStore, create_vector_store
from velorag.retrieval.hybrid import HybridRetriever
from velorag.retrieval.reranker import BaseReranker, create_reranker
from velorag.utils.cache import MemoryCache


class VeloRAG:
    def __init__(self, settings: Settings | None = None, config: Any = None):
        if settings is None and config is not None:
            settings = self._settings_from_config(config)
        elif settings is not None and not isinstance(settings, Settings):
            settings = self._settings_from_config(settings)

        self.settings = settings or Settings()
        self.embedder: BaseEmbedder = create_embedder(self.settings.embedding)

        if self.settings.index.num_shards > 1:
            self.vector_store: BaseVectorStore = ShardedIndex(
                self.settings.index,
                vector_store_factory=lambda s: create_vector_store(s),
            )
        else:
            self.vector_store = create_vector_store(self.settings.index)

        self.knowledge_graph = (
            KnowledgeGraph(self.settings.graph)
            if self.settings.graph.enabled
            else KnowledgeGraph(self.settings.graph)
        )
        self.loader = DocumentLoader(self.settings.chunk)
        self.retriever = HybridRetriever(
            self.settings.retrieval,
            self.vector_store,
            self.knowledge_graph if self.settings.graph.enabled else None,
        )
        self.reranker: BaseReranker | None = create_reranker(self.settings.retrieval.reranker)

        self.generator = LLMGenerator()
        self.cache = MemoryCache(max_entries=self.settings.cache.max_entries)
        self._all_chunks: list[Chunk] = []

    @staticmethod
    def _settings_from_config(cfg: Any) -> Settings:
        s = Settings()
        if hasattr(cfg, "chunking"):
            s.chunk.strategy = cfg.chunking.strategy
            s.chunk.chunk_size = cfg.chunking.chunk_size
            s.chunk.chunk_overlap = cfg.chunking.chunk_overlap
        if hasattr(cfg, "embedding"):
            s.embedding.backend = cfg.embedding.backend
            s.embedding.model_name = cfg.embedding.model_name
            s.embedding.dimension = cfg.embedding.dimension
        if hasattr(cfg, "vector_store"):
            s.index.backend = cfg.vector_store.backend
            s.index.collection_name = cfg.vector_store.collection_name
        if hasattr(cfg, "reranker") and cfg.reranker.enabled:
            s.retrieval.reranker = cfg.reranker.model_type
        if hasattr(cfg, "knowledge_graph"):
            s.graph.enabled = cfg.knowledge_graph.enabled
            s.graph.max_entity_distance = cfg.knowledge_graph.max_hops
        return s

    @property
    def document_count(self) -> int:
        return len(self._all_chunks)

    def ingest(self, path: str | Path) -> int:
        chunks = self.ingest_file(str(path))
        return len(chunks)

    def ingest_text(
        self,
        text: str,
        doc_id: str = "inline_doc",
        source: str = "inline",
        metadata: dict | None = None,
    ) -> int:
        meta = {"source": source, **(metadata or {})}
        doc = Document(id=doc_id, content=text, metadata=meta, source=source)
        chunker = Chunker(self.settings.chunk)
        chunks = chunker.chunk(doc)
        self._index_chunks(chunks)
        return len(chunks)

    def ingest_file(self, file_path: str) -> list[Chunk]:
        p = Path(file_path)
        if p.is_dir():
            return self.ingest_directory(file_path)
        chunks = self.loader.load_file_sync(file_path)
        self._index_chunks(chunks)
        return chunks

    def ingest_file_sync(self, file_path: str) -> list[Chunk]:
        return self.ingest_file(file_path)

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> list[Chunk]:
        docs = self.loader.load(dir_path)
        chunks = self.loader.chunker.chunk_documents(docs)
        self._index_chunks(chunks)
        return chunks

    def _index_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        if self.knowledge_graph and self.settings.graph.enabled:
            self.knowledge_graph.extract_from_chunks(chunks)

        self.vector_store.add(chunks, embeddings)
        self._all_chunks.extend(chunks)
        self.retriever.set_chunks(self._all_chunks)
        self.cache.clear()

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        use_graph: bool = True,
        filters: dict | None = None,
        candidate_k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve candidates, rerank them, then select the final top_k results.

        candidate_k limits the fused candidate pool, independently of top_k.
        Individual retrieval sources retain their configured search depths.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if candidate_k is None:
            candidate_k = max(
                top_k,
                self.settings.retrieval.dense_top_k if self.reranker else top_k,
                self.settings.retrieval.sparse_top_k if self.reranker else top_k,
            )
        if candidate_k < top_k:
            raise ValueError("candidate_k must be at least top_k")
        actual_mode = "hybrid" if mode == "graph_enhanced" else mode
        actual_graph = True if mode == "graph_enhanced" else use_graph

        cache_key = f"search:{query}:{top_k}:{candidate_k}:{actual_mode}:{actual_graph}:{filters}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        query_embedding = self.embedder.embed_query(query)

        context = QueryContext(
            query=query,
            top_k=candidate_k,
            mode=RetrievalMode(actual_mode),
            use_graph=actual_graph,
            filters=filters or {},
        )

        results = self.retriever.retrieve(query_embedding, context)

        if self.reranker:
            results = self.reranker.rerank(query, results)

        if self.settings.retrieval.min_score_threshold > 0:
            results = [r for r in results if r.score >= self.settings.retrieval.min_score_threshold]

        results = results[:top_k]
        self.cache.set(cache_key, results)
        return results

    def query(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        use_graph: bool = True,
        system_prompt: Optional[str] = None,
        min_relevance_score: float = 0.12,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> GenerationResponse:
        results = self.search(query=query, top_k=top_k, mode=mode, use_graph=use_graph)

        request = GenerationRequest(
            query=query,
            context_chunks=[r.chunk for r in results],
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            min_relevance_score=min_relevance_score,
            stream=stream,
        )

        return self.generator.generate(request, results)

    async def query_stream(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        results = self.search(query=query, top_k=top_k, mode=mode)

        request = GenerationRequest(
            query=query,
            context_chunks=[r.chunk for r in results],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for token in self.generator.generate_stream(request, results):
            yield token

    def get_chunks(self, doc_id: str | None = None) -> list[Chunk]:
        if doc_id:
            return [c for c in self._all_chunks if c.doc_id == doc_id]
        return self._all_chunks
