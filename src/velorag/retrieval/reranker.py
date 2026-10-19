from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from velorag.core.types import SearchResult, RerankerType
from velorag.retrieval.engine import tokenize_multilingual


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        ...


class LexicalOverlapReranker(BaseReranker):
    """Fast multilingual lexical + bigram overlap reranker that requires no external GPU models."""

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results

        q_tokens = set(tokenize_multilingual(query))
        if not q_tokens:
            return results

        reranked: list[SearchResult] = []
        for r in results:
            c_tokens = set(tokenize_multilingual(r.chunk.text))
            overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            exact_bonus = 0.15 if query.strip().lower() in r.chunk.text.lower() else 0.0
            new_score = self.alpha * r.score + (1.0 - self.alpha) * min(1.0, overlap + exact_bonus)
            reranked.append(
                SearchResult(
                    chunk=r.chunk,
                    score=round(float(new_score), 4),
                    retrieval_source=r.retrieval_source,
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback = LexicalOverlapReranker()

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = False
        return self._model

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results

        m = self.model
        if not m:
            return self._fallback.rerank(query, results)

        pairs = [(query, r.chunk.text) for r in results]
        scores = m.predict(pairs)

        for result, score in zip(results, scores):
            result.score = float(score)

        results.sort(key=lambda x: x.score, reverse=True)
        return results


def create_reranker(reranker_type: Optional[RerankerType]) -> Optional[BaseReranker]:
    if reranker_type == RerankerType.CROSS_ENCODER:
        return CrossEncoderReranker()
    if reranker_type == RerankerType.LEXICAL_OVERLAP:
        return LexicalOverlapReranker()
    return None
