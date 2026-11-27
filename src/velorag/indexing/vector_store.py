from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from velorag.core.settings import IndexSettings
from velorag.core.types import Chunk, SearchResult
from velorag.retrieval.engine import matches_filters


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]):
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class InMemoryVectorStore(BaseVectorStore):
    """线程安全的内存向量存储，支持先过滤后截断（避免 Filter-after-TopK 漏召回）与幂等更新。"""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: list[np.ndarray] = []
        self._ids: list[str] = []
        self._id_to_pos: dict[str, int] = {}
        self._lock = threading.RLock()

    @property
    def dimension(self) -> int:
        return self._dimension

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks 数量 ({len(chunks)}) 与 embeddings 数量 ({len(embeddings)}) 不一致"
            )
        with self._lock:
            for chunk, emb in zip(chunks, embeddings):
                vec = np.array(emb, dtype=np.float32)
                self._chunks[chunk.id] = chunk
                if chunk.id in self._id_to_pos:
                    pos = self._id_to_pos[chunk.id]
                    self._embeddings[pos] = vec
                else:
                    self._id_to_pos[chunk.id] = len(self._ids)
                    self._ids.append(chunk.id)
                    self._embeddings.append(vec)
                self._dimension = len(emb)

    def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if top_k <= 0 or not query_embedding:
            return []

        with self._lock:
            if not self._embeddings:
                return []

            query = np.array(query_embedding, dtype=np.float32)
            query_norm = float(np.linalg.norm(query))
            if query_norm <= 1e-10:
                return []
            query = query / query_norm

            # 先根据过滤条件筛选候选，避免全库 Top-K 截断导致目标知识库结果丢失
            if filters:
                candidate_indices = [
                    idx
                    for idx, cid in enumerate(self._ids)
                    if matches_filters(self._chunks[cid], filters)
                ]
                if not candidate_indices:
                    return []
                stacked = np.stack([self._embeddings[i] for i in candidate_indices])
                candidate_ids = [self._ids[i] for i in candidate_indices]
            else:
                stacked = np.stack(self._embeddings)
                candidate_ids = list(self._ids)

            norms = np.linalg.norm(stacked, axis=1, keepdims=True)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            normalized = stacked / safe_norms
            scores = np.dot(normalized, query)
            # 零向量文档余弦得分置 0
            zero_mask = norms.squeeze(-1) <= 1e-10
            if np.any(zero_mask):
                scores = np.where(zero_mask, 0.0, scores)

            ranked_indices = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(
                    chunk=self._chunks[candidate_ids[int(idx)]],
                    score=float(scores[int(idx)]),
                    retrieval_source="dense",
                )
                for idx in ranked_indices
            ]

    def delete(self, chunk_ids: list[str]):
        if not chunk_ids:
            return
        delete_set = set(chunk_ids)
        with self._lock:
            keep_indices = [i for i, cid in enumerate(self._ids) if cid not in delete_set]
            self._ids = [self._ids[i] for i in keep_indices]
            self._embeddings = [self._embeddings[i] for i in keep_indices]
            self._id_to_pos = {cid: i for i, cid in enumerate(self._ids)}
            for cid in delete_set:
                self._chunks.pop(cid, None)

    def count(self) -> int:
        with self._lock:
            return len(self._ids)


def _to_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """将统一过滤条件转换为 ChromaDB 兼容的 where 表达式。"""
    if not filters:
        return None
    clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        if key == "query":
            continue
        clauses.append({key: value})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, settings: IndexSettings):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        self._client = chromadb.PersistentClient(
            path=settings.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def dimension(self) -> int:
        return 384

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in chunks],
        )

    def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if top_k <= 0 or not query_embedding:
            return []
        total = self._collection.count()
        if total == 0:
            return []

        where_filter = _to_chroma_where(filters)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                metadata = dict(results["metadatas"][0][i]) if results["metadatas"] else {}
                doc_id = str(metadata.pop("doc_id", ""))
                chunk = Chunk(
                    id=cid,
                    text=results["documents"][0][i] if results["documents"] else "",
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata=metadata,
                )
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - min(float(distance), 1.0)
                search_results.append(
                    SearchResult(chunk=chunk, score=score, retrieval_source="dense")
                )

        return search_results

    def delete(self, chunk_ids: list[str]):
        if not chunk_ids:
            return
        self._collection.delete(ids=chunk_ids)

    def count(self) -> int:
        return self._collection.count()


def create_vector_store(settings: IndexSettings) -> BaseVectorStore:
    from velorag.core.types import VectorStoreBackend

    if settings.backend == VectorStoreBackend.CHROMA:
        try:
            return ChromaVectorStore(settings)
        except Exception:
            return InMemoryVectorStore()
    return InMemoryVectorStore()
