from __future__ import annotations

import json
import threading
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
from velorag.retrieval.engine import matches_filters
from velorag.retrieval.hybrid import HybridRetriever
from velorag.retrieval.reranker import BaseReranker, create_reranker
from velorag.utils.cache import MemoryCache


class VeloRAG:
    """线程安全的端到端混合检索与可验证引用生成管线。"""

    def __init__(self, settings: Settings | None = None, config: Any = None):
        raw_config = config
        if settings is None and config is not None:
            settings = self._settings_from_config(config)
        elif settings is not None and not isinstance(settings, Settings):
            raw_config = settings
            settings = self._settings_from_config(settings)

        self.settings = settings or Settings()
        self._lock = threading.RLock()
        self.embedder: BaseEmbedder = create_embedder(self.settings.embedding)

        if self.settings.index.num_shards > 1:
            self.vector_store: BaseVectorStore = ShardedIndex(
                self.settings.index,
                vector_store_factory=lambda s: create_vector_store(s),
            )
        else:
            self.vector_store = create_vector_store(self.settings.index)

        self.knowledge_graph = KnowledgeGraph(self.settings.graph)
        self.loader = DocumentLoader(self.settings.chunk)
        self.retriever = HybridRetriever(
            self.settings.retrieval,
            self.vector_store,
            self.knowledge_graph if self.settings.graph.enabled else None,
        )
        self.reranker: BaseReranker | None = create_reranker(self.settings.retrieval.reranker)

        gen_cfg = getattr(raw_config, "generation", None)
        if gen_cfg is not None:
            self.generator = LLMGenerator(
                api_base=getattr(gen_cfg, "api_base", None),
                api_key=getattr(gen_cfg, "api_key", None),
                model=getattr(gen_cfg, "model_name", None),
                system_prompt=getattr(gen_cfg, "system_prompt", None),
            )
        else:
            self.generator = LLMGenerator()

        self.cache = MemoryCache(
            max_entries=self.settings.cache.max_entries,
            ttl_seconds=self.settings.cache.ttl_seconds,
        )
        self._all_chunks: list[Chunk] = []

    @staticmethod
    def _settings_from_config(cfg: Any) -> Settings:
        from velorag.core.settings import get_settings

        # 复用当前已加载的安全与运行环境配置，避免在生产环境中回退到默认弱密钥
        active = get_settings()
        s = Settings(auth=active.auth, db=active.db, server=active.server)
        if hasattr(cfg, "chunking"):
            s.chunk.strategy = cfg.chunking.strategy
            s.chunk.chunk_size = cfg.chunking.chunk_size
            s.chunk.chunk_overlap = cfg.chunking.chunk_overlap
            if hasattr(cfg.chunking, "separators"):
                s.chunk.separators = list(cfg.chunking.separators)
        if hasattr(cfg, "embedding"):
            s.embedding.backend = cfg.embedding.backend
            s.embedding.model_name = cfg.embedding.model_name
            s.embedding.dimension = cfg.embedding.dimension
            s.embedding.batch_size = getattr(cfg.embedding, "batch_size", s.embedding.batch_size)
            s.embedding.api_key = getattr(cfg.embedding, "api_key", None)
            s.embedding.api_base = getattr(cfg.embedding, "api_base", None)
        if hasattr(cfg, "vector_store"):
            s.index.backend = cfg.vector_store.backend
            s.index.collection_name = cfg.vector_store.collection_name
            if getattr(cfg.vector_store, "persist_directory", None):
                s.index.persist_dir = cfg.vector_store.persist_directory
        if hasattr(cfg, "retrieval"):
            s.retrieval.hybrid_top_k = getattr(cfg.retrieval, "top_k", s.retrieval.hybrid_top_k)
            dense_w = getattr(cfg.retrieval, "dense_weight", None)
            sparse_w = getattr(cfg.retrieval, "sparse_weight", None)
            if dense_w is not None and sparse_w is not None and (dense_w + sparse_w) > 0:
                s.retrieval.fusion_alpha = float(dense_w) / float(dense_w + sparse_w)
        if hasattr(cfg, "reranker") and cfg.reranker.enabled:
            s.retrieval.reranker = cfg.reranker.model_type
        if hasattr(cfg, "knowledge_graph"):
            s.graph.enabled = cfg.knowledge_graph.enabled
            s.graph.max_entity_distance = cfg.knowledge_graph.max_hops
        return s

    @property
    def document_count(self) -> int:
        with self._lock:
            return len(self._all_chunks)

    def ingest(self, path: str | Path, metadata: dict[str, Any] | None = None) -> int:
        chunks = self.ingest_file(str(path), metadata=metadata)
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

    def ingest_file(
        self, file_path: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        p = Path(file_path)
        if p.is_dir():
            return self.ingest_directory(file_path, metadata=metadata)
        chunks = self.loader.load_file_sync(file_path, metadata=metadata)
        self._index_chunks(chunks)
        return chunks

    def ingest_file_sync(
        self, file_path: str, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        return self.ingest_file(file_path, metadata=metadata)

    def ingest_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        docs = self.loader.load(dir_path, recursive=recursive, metadata=metadata)
        chunks = self.loader.chunker.chunk_documents(docs)
        self._index_chunks(chunks)
        return chunks

    def _index_chunks(self, chunks: list[Chunk]):
        if not chunks:
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        with self._lock:
            if self.knowledge_graph and self.settings.graph.enabled:
                self.knowledge_graph.extract_from_chunks(chunks)

            self.vector_store.add(chunks, embeddings)
            self._all_chunks.extend(chunks)
            self.retriever.set_chunks(self._all_chunks)
            self.cache.clear()

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """删除指定 chunk_ids 并同步清理向量库、BM25、知识图谱与查询缓存。"""
        remove_set = set(chunk_ids)
        if not remove_set:
            return 0
        with self._lock:
            before = len(self._all_chunks)
            self._all_chunks = [c for c in self._all_chunks if c.id not in remove_set]
            removed = before - len(self._all_chunks)
            if removed == 0:
                return 0
            self.vector_store.delete(list(remove_set))
            if self.knowledge_graph:
                self.knowledge_graph.remove_chunks(remove_set)
            self.retriever.set_chunks(self._all_chunks)
            self.cache.clear()
            return removed

    def delete_by_kb(self, kb_id: str) -> int:
        """删除指定知识库下的全部索引分块与图谱关联。"""
        with self._lock:
            target_ids = [c.id for c in self._all_chunks if c.metadata.get("kb_id") == kb_id]
        return self.delete_chunks(target_ids)

    def delete_by_doc(self, doc_id: str) -> int:
        """删除指定文档下的全部索引分块与图谱关联。"""
        with self._lock:
            target_ids = [c.id for c in self._all_chunks if c.doc_id == doc_id]
        return self.delete_chunks(target_ids)

    def get_chunk_ids_for_filters(self, filters: dict[str, Any] | None) -> Optional[set[str]]:
        """根据过滤条件计算允许访问的 chunk_id 集合，用于知识图谱子图多租户隔离。"""
        if not filters:
            return None
        with self._lock:
            return {c.id for c in self._all_chunks if matches_filters(c, filters)}

    @staticmethod
    def _build_cache_key(
        query: str,
        top_k: int,
        mode: str,
        use_graph: bool,
        filters: dict | None,
    ) -> str:
        payload = {
            "q": query,
            "k": top_k,
            "m": mode,
            "g": use_graph,
            "f": filters or {},
        }
        return "search:" + json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        use_graph: bool = True,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or not query or not query.strip():
            return []

        actual_mode = "hybrid" if mode == "graph_enhanced" else mode
        actual_graph = True if mode == "graph_enhanced" else use_graph

        cache_key = self._build_cache_key(query, top_k, actual_mode, actual_graph, filters)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return [
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retrieval_source=r.retrieval_source,
                )
                for r in cached
            ]

        query_embedding = self.embedder.embed_query(query)
        context = QueryContext(
            query=query,
            top_k=top_k,
            mode=RetrievalMode(actual_mode),
            use_graph=actual_graph,
            filters=dict(filters) if filters else {},
        )

        with self._lock:
            results = self.retriever.retrieve(query_embedding, context)

        if self.reranker:
            results = self.reranker.rerank(query, results)

        if self.settings.retrieval.min_score_threshold > 0:
            results = [r for r in results if r.score >= self.settings.retrieval.min_score_threshold]

        results = results[:top_k]
        self.cache.set(
            cache_key,
            [
                SearchResult(
                    chunk=r.chunk,
                    score=r.score,
                    retrieval_source=r.retrieval_source,
                )
                for r in results
            ],
        )
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
        filters: dict | None = None,
    ) -> GenerationResponse:
        results = self.search(
            query=query,
            top_k=top_k,
            mode=mode,
            use_graph=use_graph,
            filters=filters,
        )

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
        use_graph: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        filters: dict | None = None,
    ) -> AsyncIterator[str]:
        results = self.search(
            query=query,
            top_k=top_k,
            mode=mode,
            use_graph=use_graph,
            filters=filters,
        )

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
        with self._lock:
            if doc_id:
                return [c for c in self._all_chunks if c.doc_id == doc_id]
            return list(self._all_chunks)
