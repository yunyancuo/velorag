from __future__ import annotations

from typing import Optional

from velorag.core.settings import RetrievalSettings
from velorag.core.types import Chunk, QueryContext, SearchResult, RetrievalMode
from velorag.graph.knowledge_graph import KnowledgeGraph
from velorag.retrieval.engine import (
    dense_retrieval,
    sparse_retrieval,
    reciprocal_rank_fusion,
    min_max_normalize,
    BM25Scorer,
)


class HybridRetriever:
    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        vector_store: object = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ):
        self.settings = settings or RetrievalSettings()
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self._bm25: BM25Scorer | None = None
        self._all_chunks: list[Chunk] = []

    def set_chunks(self, chunks: list[Chunk]):
        self._all_chunks = chunks
        self._bm25 = BM25Scorer()
        self._bm25.index(chunks)

    def retrieve(self, query_embedding: list[float], context: QueryContext) -> list[SearchResult]:
        all_results: list[list[SearchResult]] = []

        if context.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID, RetrievalMode.GRAPH_ENHANCED):
            if self.vector_store:
                dense_results = dense_retrieval(
                    query_embedding,
                    self.settings.dense_top_k,
                    self.vector_store,
                    context.filters,
                )
                all_results.append(min_max_normalize(dense_results))

        if context.mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID, RetrievalMode.GRAPH_ENHANCED):
            sparse_results = sparse_retrieval(
                context.query,
                self.settings.sparse_top_k,
                self._all_chunks,
                self._bm25,
            )
            all_results.append(min_max_normalize(sparse_results))

        if context.mode == RetrievalMode.GRAPH_ENHANCED and context.use_graph and self.knowledge_graph:
            graph_results = self._graph_enhanced_retrieval(context.query, query_embedding)
            if graph_results:
                all_results.append(min_max_normalize(graph_results))

        if not all_results:
            return []

        fused = reciprocal_rank_fusion(all_results)
        return fused[:context.top_k]

    def _graph_enhanced_retrieval(
        self, query: str, query_embedding: list[float]
    ) -> list[SearchResult]:
        if not self.knowledge_graph:
            return []

        entities = self.knowledge_graph.find_relevant_entities(query)
        if not entities:
            return []

        all_chunk_ids: set[str] = set()
        for entity in entities:
            chunk_ids = self.knowledge_graph.get_chunks_for_entity(
                entity.id, self.knowledge_graph.settings.max_entity_distance
            )
            all_chunk_ids.update(chunk_ids)

        if not all_chunk_ids:
            return []

        chunk_map = {c.id: c for c in self._all_chunks}
        results = []
        for cid in all_chunk_ids:
            if cid in chunk_map:
                results.append(SearchResult(
                    chunk=chunk_map[cid],
                    score=0.8,
                    retrieval_source="graph",
                ))

        return results
