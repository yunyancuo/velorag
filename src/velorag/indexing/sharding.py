from __future__ import annotations

import zlib
from typing import Callable, Optional

from velorag.core.settings import IndexSettings
from velorag.core.types import Chunk, SearchResult
from velorag.indexing.vector_store import BaseVectorStore, InMemoryVectorStore


class InMemoryShard(InMemoryVectorStore):
    """基于 InMemoryVectorStore 的分片实现，完整继承过滤、并发安全与幂等更新能力。"""


class ShardedIndex(BaseVectorStore):
    """基于确定性 CRC32 哈希路由的多分片向量索引。"""

    def __init__(
        self,
        settings: IndexSettings,
        vector_store_factory: Optional[Callable[[IndexSettings], BaseVectorStore]] = None,
    ):
        self.settings = settings
        self.num_shards = max(1, settings.num_shards)
        self._shards: list[BaseVectorStore] = []
        self._factory = vector_store_factory

    @property
    def shards(self) -> list[BaseVectorStore]:
        if not self._shards:
            for _ in range(self.num_shards):
                store = self._factory(self.settings) if self._factory else InMemoryShard()
                self._shards.append(store)
        return self._shards

    def _shard_index(self, chunk_id: str) -> int:
        return zlib.crc32(chunk_id.encode("utf-8")) % self.num_shards

    def _get_shard(self, chunk_id: str) -> BaseVectorStore:
        return self.shards[self._shard_index(chunk_id)]

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]):
        shard_groups: dict[int, tuple[list[Chunk], list[list[float]]]] = {}
        for chunk, emb in zip(chunks, embeddings):
            shard_idx = self._shard_index(chunk.id)
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = ([], [])
            shard_groups[shard_idx][0].append(chunk)
            shard_groups[shard_idx][1].append(emb)

        for shard_idx, (c, e) in shard_groups.items():
            self.shards[shard_idx].add(c, e)

    def search(
        self, query_embedding: list[float], top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        per_shard_k = max(top_k, min(100, top_k * 2))

        all_results: list[SearchResult] = []
        for shard in self.shards:
            results = shard.search(query_embedding, per_shard_k, filters)
            all_results.extend(results)

        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]

    def delete(self, chunk_ids: list[str]):
        if not chunk_ids:
            return
        grouped: dict[int, list[str]] = {}
        for chunk_id in chunk_ids:
            grouped.setdefault(self._shard_index(chunk_id), []).append(chunk_id)
        for shard_idx, cids in grouped.items():
            self.shards[shard_idx].delete(cids)

    def count(self) -> int:
        return sum(s.count() for s in self.shards if s)
