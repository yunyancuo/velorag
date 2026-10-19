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
        weights: list[float] = []

        if context.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID, RetrievalMode.GRAPH_ENHANCED):
            if self.vector_store:
                dense_results = dense_retrieval(
                    query_embedding,
                    self.settings.dense_top_k,
                    self.vector_store,
                    context.filters,
                )
                if dense_results:
                    all_results.append(min_max_normalize(dense_results))
                    weights.append(self.settings.fusion_alpha)

        if context.mode in (RetrievalMode.SPARSE, RetrievalMode.HYBRID, RetrievalMode.GRAPH_ENHANCED):
            sparse_results = sparse_retrieval(
                context.query,
                self.settings.sparse_top_k,
                self._all_chunks,
                self._bm25,
                filters=context.filters,
            )
            if sparse_results:
                all_results.append(min_max_normalize(sparse_results))
                weights.append(1.0 - self.settings.fusion_alpha)

        if (
            context.mode in (RetrievalMode.HYBRID, RetrievalMode.GRAPH_ENHANCED)
            and context.use_graph
            and self.knowledge_graph
        ):
            graph_results = self._graph_enhanced_retrieval(context.query, context.filters)
            if graph_results:
                all_results.append(min_max_normalize(graph_results))
                weights.append(0.35)

        if not all_results:
            return []

        if len(all_results) == 1:
            return all_results[0][: context.top_k]

        fused = reciprocal_rank_fusion(all_results, weights=weights)
        return fused[: context.top_k]

    def _graph_enhanced_retrieval(
        self, query: str, filters: dict | None = None
    ) -> list[SearchResult]:
        if not self.knowledge_graph:
            return []

        entities = self.knowledge_graph.find_relevant_entities(query)
        if not entities:
            return []

        chunk_best_score: dict[str, float] = {}
        for entity in entities:
            dist_map = self.knowledge_graph.get_chunks_with_distance(
                entity.id, self.knowledge_graph.settings.max_entity_distance
            )
            for cid, hop in dist_map.items():
                hop_score = 1.0 / (1.0 + 0.35 * hop)
                chunk_best_score[cid] = chunk_best_score.get(cid, 0.0) + hop_score

        if not chunk_best_score:
            return []

        chunk_map = {c.id: c for c in self._all_chunks}
        results: list[SearchResult] = []
        for cid, score in sorted(chunk_best_score.items(), key=lambda x: x[1], reverse=True):
            if cid not in chunk_map:
                continue
            chunk = chunk_map[cid]
            if filters:
                if any(
                    (chunk.doc_id if key == "doc_id" else chunk.metadata.get(key)) != value
                    for key, value in filters.items()
                    if key != "query"
                ):
                    continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    retrieval_source="graph",
                )
            )

        return results
