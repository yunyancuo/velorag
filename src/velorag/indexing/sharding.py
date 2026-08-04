from __future__ import annotations

import math
import uuid
from typing import Optional

import numpy as np

from velorag.core.settings import IndexSettings
from velorag.core.types import Chunk, SearchResult
from velorag.indexing.vector_store import BaseVectorStore


class ShardedIndex:
    def __init__(
        self,
        settings: IndexSettings,
        vector_store_factory: callable = None,
    ):
        self.settings = settings
        self.num_shards = max(1, settings.num_shards)
        self._shards: list[BaseVectorStore] = []
        self._factory = vector_store_factory

    @property
    def shards(self) -> list[BaseVectorStore]:
        if not self._shards:
            for i in range(self.num_shards):
                store = self._factory(self.settings) if self._factory else InMemoryShard()
                self._shards.append(store)
        return self._shards

    def _get_shard(self, chunk_id: str) -> BaseVectorStore:
        idx = hash(chunk_id) % self.num_shards
        return self.shards[idx]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        shard_groups: dict[int, tuple[list[Chunk], list[list[float]]]] = {}
        for chunk, emb in zip(chunks, embeddings):
            shard_idx = hash(chunk.id) % self.num_shards
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = ([], [])
            shard_groups[shard_idx][0].append(chunk)
            shard_groups[shard_idx][1].append(emb)

        for shard_idx, (c, e) in shard_groups.items():
            self.shards[shard_idx].add(c, e)

    def search(self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None) -> list[SearchResult]:
        per_shard_k = max(top_k, min(100, top_k * 2))

        all_results: list[SearchResult] = []
        for shard in self.shards:
            results = shard.search(query_embedding, per_shard_k, filters)
            all_results.extend(results)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def delete(self, chunk_ids: list[str]):
        for chunk_id in chunk_ids:
            self._get_shard(chunk_id).delete([chunk_id])

    def count(self) -> int:
        return sum(s.count() for s in self.shards if s)


class InMemoryShard(BaseVectorStore):
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: list[np.ndarray] = []
        self._ids: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

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
            chunk_id = self._ids[idx]
            results.append(SearchResult(
                chunk=self._chunks[chunk_id],
                score=float(scores[idx]),
                retrieval_source="dense",
            ))
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
