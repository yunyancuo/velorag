from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: set[str], k: int = 5) -> float:
    """Compute Recall@K over unique document IDs."""
    if not relevant_doc_ids:
        return 1.0
    top_k = list(dict.fromkeys(retrieved_doc_ids))[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def mrr_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: set[str], k: int = 5) -> float:
    """Compute Mean Reciprocal Rank (MRR@K)."""
    if not relevant_doc_ids:
        return 1.0
    top_k = list(dict.fromkeys(retrieved_doc_ids))[:k]
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: Sequence[str], relevant_doc_ids: set[str], k: int = 5) -> float:
    """Compute binary Normalized Discounted Cumulative Gain (NDCG@K)."""
    if not relevant_doc_ids:
        return 1.0
    top_k = list(dict.fromkeys(retrieved_doc_ids))[:k]
    dcg = 0.0
    for idx, doc_id in enumerate(top_k):
        if doc_id in relevant_doc_ids:
            dcg += 1.0 / math.log2(idx + 2)

    ideal_hits = min(len(relevant_doc_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def citation_precision(cited_doc_ids: Sequence[str], relevant_doc_ids: set[str]) -> float:
    """Fraction of cited sources that belong to ground-truth relevant documents."""
    if not cited_doc_ids:
        return 0.0
    hits = sum(1 for doc_id in cited_doc_ids if doc_id in relevant_doc_ids)
    return hits / len(cited_doc_ids)


def keyword_coverage(answer: str, expected_keywords: Sequence[str]) -> float:
    """Fraction of expected domain keywords present in the generated answer."""
    if not expected_keywords:
        return 1.0
    lower_ans = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower_ans)
    return hits / len(expected_keywords)
