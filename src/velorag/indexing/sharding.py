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

        # Legacy persistent shard factories can alias the same collection.
        unique = {}
        for result in all_results:
            if result.chunk.id not in unique or result.score > unique[result.chunk.id].score:
                unique[result.chunk.id] = result
        return sorted(unique.values(), key=lambda x: x.score, reverse=True)[:top_k]

    def delete(self, chunk_ids: list[str]):
        for chunk_id in chunk_ids:
            self._get_shard(chunk_id).delete([chunk_id])

    def count(self) -> int:
        return sum(s.count() for s in self.shards if s)

    def snapshot(self):
        records = {}
        for shard in self.shards:
            shard_chunks, shard_embeddings = shard.snapshot()
            if len(shard_chunks) != len(shard_embeddings):
                raise ValueError("Shard snapshot chunk/embedding counts differ")
            for chunk, embedding in zip(shard_chunks, shard_embeddings):
                if chunk.id in records:
                    old_chunk, old_embedding = records[chunk.id]
                    if old_chunk != chunk or not np.array_equal(old_embedding, embedding):
                        raise ValueError(f"Conflicting duplicate chunk in shard snapshots: {chunk.id}")
                else:
                    records[chunk.id] = (chunk, embedding)
        return ([r[0] for r in records.values()], [r[1] for r in records.values()])


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

    def snapshot(self):
        latest = {cid: i for i, cid in enumerate(self._ids)}
        return ([self._chunks[cid] for cid in latest],
                [self._embeddings[i].tolist() for i in latest.values()])
