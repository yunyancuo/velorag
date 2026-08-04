from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import numpy as np

from velorag.core.settings import RetrievalSettings
from velorag.core.types import Chunk, QueryContext, SearchResult, RetrievalMode


def dense_retrieval(
    query_embedding: list[float],
    top_k: int,
    vector_store: object,
    filters: dict | None = None,
) -> list[SearchResult]:
    return vector_store.search(query_embedding, top_k=top_k, filters=filters)


def sparse_retrieval(
    query: str,
    top_k: int,
    chunks: list[Chunk],
    bm25_index: object = None,
) -> list[SearchResult]:
    if bm25_index is None:
        bm25_index = BM25Scorer()
        bm25_index.index(chunks)

    return bm25_index.search(query, top_k)


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freq: Counter = Counter()
        self._doc_len: list[int] = []
        self._avg_dl: float = 0.0
        self._N: int = 0

    def index(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._N = len(chunks)
        self._doc_len = []
        self._doc_freq = Counter()

        tokenized_docs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tokenized_docs.append(tokens)
            self._doc_len.append(len(tokens))
            self._doc_freq.update(set(tokens))

        self._avg_dl = sum(self._doc_len) / max(1, self._N)

    def _tokenize(self, text: str) -> list[str]:
        import re
        return [t.lower() for t in re.findall(r'\w+', text)]

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        if self._N == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = np.zeros(self._N)

        for token in query_tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            for i, chunk in enumerate(self._chunks):
                tf = chunk.text.lower().count(token)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_dl)
                    scores[i] += idf * numerator / denominator

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(SearchResult(
                    chunk=self._chunks[idx],
                    score=float(scores[idx]),
                    retrieval_source="sparse",
                ))

        return results


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, tuple[float, SearchResult]] = {}

    for results in result_sets:
        for rank, result in enumerate(results):
            chunk_id = result.chunk.id
            rrf_score = 1.0 / (k + rank + 1)
            if chunk_id in scores:
                scores[chunk_id] = (scores[chunk_id][0] + rrf_score, result)
            else:
                scores[chunk_id] = (rrf_score, result)

    fused = list(scores.values())
    fused.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in fused]


def min_max_normalize(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results
    scores = [r.score for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r.score = 1.0
        return results
    for r in results:
        r.score = (r.score - min_s) / (max_s - min_s)
    return results
