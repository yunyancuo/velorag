from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from velorag.core.settings import IndexSettings
from velorag.core.types import Chunk, SearchResult


class BaseVectorStore(ABC):
    def snapshot(self) -> tuple[list[Chunk], list[list[float]]]:
        """Export aligned chunks/vectors for opt-in derived indexes."""
        raise NotImplementedError("This store cannot export its persisted vectors")

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[SearchResult]:
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]):
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: list[np.ndarray] = []
        self._ids: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def snapshot(self):
        latest = {cid: i for i, cid in enumerate(self._ids)}
        return ([self._chunks[cid] for cid in latest],
                [self._embeddings[i].tolist() for i in latest.values()])

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        for chunk, emb in zip(chunks, embeddings):
            self._chunks[chunk.id] = chunk
            self._ids.append(chunk.id)
            self._embeddings.append(np.array(emb, dtype=np.float32))
            self._dimension = len(emb)

    def search(self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[SearchResult]:
        if not self._embeddings:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query = query / (np.linalg.norm(query) + 1e-10)
        stacked = np.stack(self._embeddings)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True) + 1e-10
        stacked = stacked / norms
        scores = np.dot(stacked, query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk_id = self._ids[idx]
            chunk = self._chunks[chunk_id]
            if filters:
                matched = True
                for k, v in filters.items():
                    if k == "query":
                        continue
                    if k == "doc_id" and chunk.doc_id != v:
                        matched = False
                        break
                    if k != "doc_id" and chunk.metadata.get(k) != v:
                        matched = False
                        break
                if not matched:
                    continue
            results.append(SearchResult(
                chunk=chunk,
                score=score,
                retrieval_source="dense",
            ))
            if len(results) >= top_k:
                break
        return results

    def delete(self, chunk_ids: list[str]):
        delete_set = set(chunk_ids)
        keep_indices = [i for i, cid in enumerate(self._ids) if cid not in delete_set]
        self._ids = [self._ids[i] for i in keep_indices]
        self._embeddings = [self._embeddings[i] for i in keep_indices]
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

    def count(self) -> int:
        return len(self._ids)


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
        self._collection.add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, **c.metadata} for c in chunks],
        )

    def search(self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[SearchResult]:
        where_filter = None
        if filters:
            where_filter = {k: v for k, v in filters.items() if k != "query"}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, cid in enumerate(results["ids"][0]):
                from velorag.core.types import Chunk
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                doc_id = metadata.pop("doc_id", "")
                chunk = Chunk(
                    id=cid,
                    text=results["documents"][0][i] if results["documents"] else "",
                    doc_id=doc_id,
                    chunk_index=0,
                    metadata=metadata,
                )
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - min(distance, 1.0)
                search_results.append(SearchResult(chunk=chunk, score=score, retrieval_source="dense"))

        return search_results

    def delete(self, chunk_ids: list[str]):
        self._collection.delete(ids=chunk_ids)

    def count(self) -> int:
        return self._collection.count()

    def snapshot(self):
        data = self._collection.get(include=["documents", "metadatas", "embeddings"])
        if not data["ids"]:
            return [], []
        if data.get("embeddings") is None:
            raise ValueError("Persisted collection did not return embeddings")
        chunks = []
        for i, cid in enumerate(data["ids"]):
            metadata = dict(data["metadatas"][i] or {})
            chunks.append(Chunk(
                id=cid, doc_id=metadata.pop("doc_id", ""),
                text=data["documents"][i] or "", metadata=metadata,
                chunk_index=int(metadata.get("chunk_index", 0)),
            ))
        return chunks, np.asarray(data["embeddings"], dtype=np.float32).tolist()


def create_vector_store(settings: IndexSettings) -> BaseVectorStore:
    from velorag.core.types import VectorStoreBackend

    if settings.backend == VectorStoreBackend.CHROMA:
        try:
            return ChromaVectorStore(settings)
        except Exception:
            return InMemoryVectorStore()
    else:
        return InMemoryVectorStore()
