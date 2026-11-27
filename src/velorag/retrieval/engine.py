from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from velorag.core.types import Chunk, SearchResult

_CJK_STOP_CHARS = frozenset("的了在是和与或及对为以于中上下有个什么吗呢请问可后内前")


def matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    """统一判断 Chunk 是否满足全部元数据过滤条件（支持精确值、$in 与 $nin）。"""
    if not filters:
        return True

    for key, expected in filters.items():
        if key == "query":
            continue

        actual = chunk.doc_id if key == "doc_id" else chunk.metadata.get(key)

        if isinstance(expected, dict):
            if "$in" in expected:
                allowed = expected["$in"]
                if actual not in allowed:
                    return False
            if "$nin" in expected:
                excluded = expected["$nin"]
                if actual in excluded:
                    return False
            # 若存在未知操作符且既非 $in 也非 $nin，则回退至精确值比较
            if "$in" not in expected and "$nin" not in expected and actual != expected:
                return False
        elif actual != expected:
            return False

    return True


def tokenize_multilingual(text: str) -> list[str]:
    """支持英文词元、数字以及中文 CJK 去停用字 unigram + bigram 的双语分词器。"""
    if not text:
        return []
    lower = text.lower()
    words = re.findall(r"[a-z0-9_]+", lower)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lower)
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
    if top_k <= 0:
        return []
    return vector_store.search(query_embedding, top_k=top_k, filters=filters)


def sparse_retrieval(
    query: str,
    top_k: int,
    chunks: list[Chunk],
    bm25_index: object = None,
    filters: dict | None = None,
) -> list[SearchResult]:
    if top_k <= 0 or not query or not query.strip():
        return []
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
        self._chunks = list(chunks)
        self._N = len(self._chunks)
        self._doc_len = []
        self._doc_freq = Counter()
        self._doc_term_freqs = []

        for chunk in self._chunks:
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
        if self._N == 0 or top_k <= 0 or not query or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = np.zeros(self._N, dtype=np.float64)
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
            chunk = self._chunks[int(idx)]
            if not matches_filters(chunk, filters):
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

    safe_k = max(1, int(k))
    if weights is None or len(weights) != len(result_sets):
        weights = [1.0] * len(result_sets)

    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}
    sources_map: dict[str, list[str]] = {}

    for weight, results in zip(weights, result_sets):
        eff_weight = max(0.0, float(weight))
        if eff_weight == 0.0:
            continue
        for rank, result in enumerate(results):
            chunk_id = result.chunk.id
            rrf_score = eff_weight * (1.0 / (safe_k + rank + 1))
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

    max_rrf = max(scores.values())
    if max_rrf <= 0:
        return []

    fused: list[SearchResult] = []
    for chunk_id, total_rrf in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        base = best_result[chunk_id]
        src_list = sources_map.get(chunk_id, [])
        source_label = (
            src_list[0]
            if len(src_list) == 1
            else f"hybrid({'+'.join(src_list)})"
        )
        # 结合归一化 RRF 排序分与单路最佳置信度，避免纯排名丢失相似度尺度
        combined_score = 0.65 * (total_rrf / max_rrf) + 0.35 * min(1.0, max(0.0, base.score))
        fused.append(
            SearchResult(
                chunk=base.chunk,
                score=round(float(combined_score), 4),
                retrieval_source=source_label,
            )
        )

    return fused


def min_max_normalize(results: list[SearchResult]) -> list[SearchResult]:
    """对检索结果进行无副作用的归一化，返回新的 SearchResult 列表。"""
    if not results:
        return []
    scores = [r.score for r in results]
    min_s, max_s = min(scores), max(scores)

    if max_s == min_s:
        normalized: list[SearchResult] = []
        for r in results:
            if max_s <= 0:
                norm_score = 0.0
            elif r.retrieval_source == "dense":
                # 单条或等分稠密向量结果保留原始余弦相似度尺度，避免无关文档被拉高到 1.0
                norm_score = min(1.0, max(0.0, float(r.score)))
            else:
                norm_score = float(r.score) / (float(r.score) + 1.0)
            normalized.append(
                SearchResult(
                    chunk=r.chunk,
                    score=round(norm_score, 4),
                    retrieval_source=r.retrieval_source,
                )
            )
        return normalized

    span = max_s - min_s
    return [
        SearchResult(
            chunk=r.chunk,
            score=float((r.score - min_s) / span),
            retrieval_source=r.retrieval_source,
        )
        for r in results
    ]
