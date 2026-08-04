from __future__ import annotations

import time
from typing import Optional, AsyncIterator

from velorag.core.settings import Settings
from velorag.core.types import (
    Chunk,
    QueryContext,
    SearchResult,
    GenerationRequest,
    GenerationResponse,
    RetrievalMode,
    RerankerType,
)
from velorag.document.loader import DocumentLoader
from velorag.embedding.embedder import create_embedder, BaseEmbedder
from velorag.indexing.vector_store import create_vector_store, BaseVectorStore, InMemoryVectorStore
from velorag.indexing.sharding import ShardedIndex
from velorag.graph.knowledge_graph import KnowledgeGraph
from velorag.retrieval.hybrid import HybridRetriever
from velorag.retrieval.reranker import BaseReranker, CrossEncoderReranker
from velorag.generation.llm import LLMGenerator
from velorag.utils.cache import MemoryCache


class VeloRAG:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.embedder: BaseEmbedder = create_embedder(self.settings.embedding)

        if self.settings.index.num_shards > 1:
            self.vector_store = ShardedIndex(
                self.settings.index,
                vector_store_factory=lambda s: create_vector_store(s),
            )
        else:
            self.vector_store = create_vector_store(self.settings.index)

        self.knowledge_graph = KnowledgeGraph(self.settings.graph) if self.settings.graph.enabled else None
        self.loader = DocumentLoader(self.settings.chunk)
        self.retriever = HybridRetriever(
            self.settings.retrieval,
            self.vector_store,
            self.knowledge_graph,
        )
        self.reranker: BaseReranker | None = None
        if self.settings.retrieval.reranker == RerankerType.CROSS_ENCODER:
            self.reranker = CrossEncoderReranker()

        self.generator = LLMGenerator()
        self.cache = MemoryCache(max_entries=self.settings.cache.max_entries)
        self._all_chunks: list[Chunk] = []

    def ingest_file(self, file_path: str) -> list[Chunk]:
        chunks = self.loader.load_file_sync(file_path)
        self._index_chunks(chunks)
        return chunks

    def ingest_file_sync(self, file_path: str) -> list[Chunk]:
        return self.ingest_file(file_path)

    def ingest_directory(self, dir_path: str, recursive: bool = True) -> list[Chunk]:
        import asyncio
        chunks = asyncio.run(self.loader.load_directory(dir_path, recursive))
        self._index_chunks(chunks)
        return chunks

    def _index_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        if self.knowledge_graph:
            self.knowledge_graph.extract_from_chunks(chunks)

        self.vector_store.add(chunks, embeddings)
        self._all_chunks.extend(chunks)
        self.retriever.set_chunks(self._all_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        use_graph: bool = True,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        cache_key = f"search:{query}:{top_k}:{mode}:{use_graph}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        query_embedding = self.embedder.embed_query(query)

        context = QueryContext(
            query=query,
            top_k=top_k,
            mode=RetrievalMode(mode),
            use_graph=use_graph,
            filters=filters or {},
        )

        results = self.retriever.retrieve(query_embedding, context)

        if self.reranker:
            results = self.reranker.rerank(query, results)

        if self.settings.retrieval.min_score_threshold > 0:
            results = [r for r in results if r.score >= self.settings.retrieval.min_score_threshold]

        self.cache.set(cache_key, results)
        return results

    def query(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> GenerationResponse:
        results = self.search(query=query, top_k=top_k, mode=mode)

        request = GenerationRequest(
            query=query,
            context_chunks=[r.chunk for r in results],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        return self.generator.generate(request, results)

    async def query_stream(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        max_tokens: int = 2048,
        temperature: float = 0.7,
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
