from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

import numpy as np

from velorag.core.settings import RetrievalSettings
from velorag.core.types import Chunk, QueryContext, SearchResult, RetrievalMode


_CJK_STOP_CHARS = frozenset("的了在是和与或及对为以于中上下有个什么吗呢请问可后内前")


def tokenize_multilingual(text: str) -> list[str]:
    """Tokenize English words, numbers, and CJK unigrams (excluding functional stop-chars) + bigrams."""
    lower = text.lower()
    words = re.findall(r"[a-z0-9_]+", lower)
    cjk_runs = re.findall(r"[一-鿿]+", lower)
    cjk_tokens: list[str] = []
    for run in cjk_runs:
        cjk_tokens.extend(ch for ch in run if ch not in _CJK_STOP_CHARS)
        if len(run) >= 2:
            cjk_tokens.extend(
                run[i : i + 2]
                for i in range(len(run) - 1)
                if not (run[i] in _CJK_STOP_CHARS and run[i + 1] in _CJK_STOP_CHARS)
            )
    return words + cjk_tokens


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
    filters: dict | None = None,
) -> list[SearchResult]:
    if bm25_index is None:
        bm25_index = BM25Scorer()
        bm25_index.index(chunks)

    return bm25_index.search(query, top_k, filters=filters)


class BM25Scorer:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_freq: Counter = Counter()
        self._doc_term_freqs: list[Counter] = []
        self._doc_len: list[int] = []
        self._avg_dl: float = 0.0
        self._N: int = 0

    def index(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._N = len(chunks)
        self._doc_len = []
        self._doc_freq = Counter()
        self._doc_term_freqs = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            tf_counter = Counter(tokens)
            self._doc_term_freqs.append(tf_counter)
            self._doc_len.append(len(tokens))
            self._doc_freq.update(tf_counter.keys())

        self._avg_dl = sum(self._doc_len) / max(1, self._N)

    def _tokenize(self, text: str) -> list[str]:
        return tokenize_multilingual(text)

    def search(
        self, query: str, top_k: int = 10, filters: dict | None = None
    ) -> list[SearchResult]:
        if self._N == 0:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self._N)
        avg_dl = max(1e-6, self._avg_dl)

        for token in query_tokens:
            df = self._doc_freq.get(token, 0)
            if df == 0:
                continue
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            for i, tf_counter in enumerate(self._doc_term_freqs):
                tf = tf_counter.get(token, 0)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (
                        1 - self.b + self.b * self._doc_len[i] / avg_dl
                    )
                    scores[i] += idf * numerator / denominator

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            chunk = self._chunks[idx]
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
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(scores[idx]),
                    retrieval_source="sparse",
                )
            )
            if len(results) >= top_k:
                break

        return results


def reciprocal_rank_fusion(
    result_sets: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    if not result_sets:
        return []

    if weights is None or len(weights) != len(result_sets):
        weights = [1.0] * len(result_sets)

    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}
    sources_map: dict[str, list[str]] = {}

    for weight, results in zip(weights, result_sets):
        for rank, result in enumerate(results):
            chunk_id = result.chunk.id
            rrf_score = weight * (1.0 / (k + rank + 1))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            if chunk_id not in best_result or result.score > best_result[chunk_id].score:
                best_result[chunk_id] = result
            src = result.retrieval_source or "unknown"
            if chunk_id not in sources_map:
                sources_map[chunk_id] = []
            if src not in sources_map[chunk_id]:
                sources_map[chunk_id].append(src)

    if not scores:
        return []

    max_rrf = max(scores.values()) or 1.0
    fused: list[SearchResult] = []
    for chunk_id, total_rrf in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        base = best_result[chunk_id]
        src_list = sources_map.get(chunk_id, [])
        source_label = (
            src_list[0]
            if len(src_list) == 1
            else f"hybrid({'+'.join(src_list)})"
        )
        # Combine normalized RRF with best normalized signal score
        combined_score = 0.65 * (total_rrf / max_rrf) + 0.35 * min(1.0, max(0.0, base.score))
        fused.append(
            SearchResult(
                chunk=base.chunk,
                score=round(float(combined_score), 4),
                retrieval_source=source_label,
            )
        )

    fused.sort(key=lambda result: result.score, reverse=True)
    return fused


def min_max_normalize(results: list[SearchResult]) -> list[SearchResult]:
    if not results:
        return results
    scores = [r.score for r in results]
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        for r in results:
            r.score = 1.0 if max_s > 0 else 0.0
        return results
    for r in results:
        r.score = (r.score - min_s) / (max_s - min_s)
    return results
