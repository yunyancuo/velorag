"""Experimental one-layer neighborhood residual retrieval.

Inspired by NEAR (Luo et al., 2026), https://arxiv.org/abs/2610.13314.
This is an independent simplified implementation: one normalized residual layer,
fixed convex fusion, exact search, and no paper-specific score-scale calibration.
It is not a full NEAR reproduction or an original algorithm claim.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np

from velorag.core.types import Chunk, SearchResult
from velorag.indexing.vector_store import BaseVectorStore


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


@dataclass(frozen=True)
class ResidualConfig:
    neighbors: int = 16
    anchors: int = 64
    query_anchors: int = 4
    weight: float = 0.25
    temperature: float = 10.0
    cluster_iterations: int = 8
    block_size: int = 256
    seed: int = 17

    def __post_init__(self):
        for key in ("neighbors", "anchors", "query_anchors", "cluster_iterations", "block_size"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("weight must be finite and in [0, 1]")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")


class ResidualIndex(BaseVectorStore):
    """Opt-in exact index; construction is quadratic, intended for bounded pilots.

    add/delete fully rebuild corpus-conditioned references. Incremental semantic
    maintenance and million-document scalability have not been implemented.
    """

    def __init__(self, config: ResidualConfig | None = None):
        self.config = config or ResidualConfig()
        self._chunks = []
        self._vectors = np.empty((0, 0), dtype=np.float32)
        self._keys = np.empty((0, 0), dtype=np.float32)
        self._anchors = np.empty((0, 0), dtype=np.float32)
        self._anchor_references = np.empty((0, 0), dtype=np.float32)
        self.build_seconds = 0.0

    @property
    def dimension(self):
        return self._vectors.shape[1]

    @property
    def chunks(self):
        return list(self._chunks)

    def count(self):
        return len(self._chunks)

    def fit(self, chunks: list[Chunk], embeddings):
        started = time.perf_counter()
        if not chunks:
            if np.asarray(embeddings).size:
                raise ValueError("Embedding count differs from chunk count")
            self.__init__(self.config)
            return
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or vectors.shape[1] == 0:
            raise ValueError("Expected one nonempty embedding vector per chunk")
        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings must be finite")
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique")

        # Stable ID order makes input permutations and tie handling reproducible.
        order = sorted(range(len(chunks)), key=lambda i: chunks[i].id)
        self._chunks = [chunks[i] for i in order]
        self._vectors = unit_rows(vectors[order])
        n = self.count()
        references = np.zeros_like(self._vectors)
        k = min(self.config.neighbors, n - 1)
        if k:
            for start in range(0, n, self.config.block_size):
                stop = min(n, start + self.config.block_size)
                similarities = self._vectors[start:stop] @ self._vectors.T
                similarities[np.arange(stop - start), np.arange(start, stop)] = -np.inf
                neighbors = np.argpartition(similarities, n - k, axis=1)[:, -k:]
                references[start:stop] = unit_rows(self._vectors[neighbors].mean(axis=1))
        projection = np.sum(self._vectors * references, axis=1, keepdims=True)
        residuals = unit_rows(self._vectors - projection * references)

        rng = np.random.default_rng(self.config.seed)
        chosen = rng.choice(n, size=min(self.config.anchors, n), replace=False)
        anchors = self._vectors[chosen].copy()
        for _ in range(self.config.cluster_iterations):
            labels = np.argmax(self._vectors @ anchors.T, axis=1)
            for i in range(len(anchors)):
                members = self._vectors[labels == i]
                if len(members):
                    anchors[i] = unit_rows(members.mean(axis=0))
        labels = np.argmax(self._vectors @ anchors.T, axis=1)
        anchor_references = references[chosen].copy()
        for i in range(len(anchors)):
            members = references[labels == i]
            if len(members):
                anchor_references[i] = unit_rows(members.mean(axis=0))
        self._anchors = anchors
        self._anchor_references = anchor_references
        self._keys = np.concatenate([
            np.sqrt(1 - self.config.weight) * self._vectors,
            np.sqrt(self.config.weight) * residuals,
        ], axis=1)
        self.build_seconds = time.perf_counter() - started

    def query_key(self, query_embedding):
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.shape != (self.dimension,) or not np.isfinite(query).all():
            raise ValueError("Query must be finite and match the index dimension")
        query = unit_rows(query)
        similarities = self._anchors @ query
        count = min(self.config.query_anchors, len(self._anchors))
        selected = np.argsort(-similarities, kind="stable")[:count]
        logits = self.config.temperature * similarities[selected]
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        reference = unit_rows(weights @ self._anchor_references[selected])
        residual = unit_rows(query - np.dot(query, reference) * reference)
        return np.concatenate([
            np.sqrt(1 - self.config.weight) * query,
            np.sqrt(self.config.weight) * residual,
        ])

    def scores(self, query_embedding):
        if not self.count():
            return np.empty(0, dtype=np.float32)
        return self._keys @ self.query_key(query_embedding)

    def search(self, query_embedding, top_k=10, filters=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.count():
            return []
        scores = self.scores(query_embedding)
        order = np.argsort(-scores, kind="stable")
        results = []
        for idx in order:
            chunk = self._chunks[idx]
            if filters and any(
                (chunk.doc_id if key == "doc_id" else chunk.metadata.get(key)) != value
                for key, value in filters.items() if key != "query"
            ):
                continue
            results.append(SearchResult(chunk, float(scores[idx]), "local_residual"))
            if len(results) == top_k:
                break
        return results

    def add(self, chunks, embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)
        if not chunks:
            if vectors.size:
                raise ValueError("Embedding count differs from chunk count")
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Expected one embedding vector per chunk")
        if self.count() and vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension changed")
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a batch")
        records = {c.id: (c, v) for c, v in zip(self._chunks, self._vectors)}
        records.update({c.id: (c, v) for c, v in zip(chunks, vectors)})
        self.fit([c for c, _ in records.values()], [v for _, v in records.values()])

    def delete(self, chunk_ids):
        excluded = set(chunk_ids)
        keep = [i for i, c in enumerate(self._chunks) if c.id not in excluded]
        self.fit([self._chunks[i] for i in keep], self._vectors[keep])

    def stats(self):
        return {
            "config": asdict(self.config), "count": self.count(),
            "dimension": self.dimension, "key_dimension": self._keys.shape[1],
            "build_seconds": self.build_seconds,
            "array_bytes": sum(v.nbytes for v in (
                self._vectors, self._keys, self._anchors, self._anchor_references
            )),
            "search_backend": "exact_inner_product",
            "update_policy": "full_rebuild",
        }
