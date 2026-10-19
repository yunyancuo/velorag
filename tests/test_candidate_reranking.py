import pytest

from velorag.core.config import EmbeddingConfig, VeloRAGConfig
from velorag.core.types import Chunk, EmbeddingBackend, SearchResult
from velorag.pipeline import VeloRAG


class ControlledRetriever:
    def __init__(self):
        self.depths = []

    def set_chunks(self, chunks):
        pass

    def retrieve(self, query_embedding, context):
        self.depths.append(context.top_k)
        return [
            SearchResult(Chunk(id=doc_id, text=doc_id, doc_id=doc_id), score=score)
            for doc_id, score in (("distractor", 0.9), ("evidence", 0.8), ("other", 0.1))
        ][:context.top_k]


class EvidenceReranker:
    def __init__(self):
        self.seen = []

    def rerank(self, query, results):
        self.seen = [r.chunk.id for r in results]
        return sorted(results, key=lambda r: r.chunk.id != "evidence")


def make_pipeline():
    rag = VeloRAG(config=VeloRAGConfig(
        embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH),
    ))
    rag.retriever = ControlledRetriever()
    rag.reranker = EvidenceReranker()
    return rag


def test_reranker_can_recover_evidence_outside_initial_top_k():
    rag = make_pipeline()
    results = rag.search("question", top_k=1)
    assert rag.retriever.depths[0] > 1
    assert "evidence" in rag.reranker.seen
    assert [r.chunk.id for r in results] == ["evidence"]


def test_candidate_budget_has_a_separate_cache_entry():
    rag = make_pipeline()
    assert rag.search("question", top_k=1, candidate_k=1)[0].chunk.id == "distractor"
    assert rag.search("question", top_k=1, candidate_k=2)[0].chunk.id == "evidence"
    rag.search("question", top_k=1, candidate_k=2)
    assert rag.retriever.depths == [1, 2]


@pytest.mark.parametrize("top_k,candidate_k", [(0, None), (-1, None), (3, 2)])
def test_invalid_candidate_budgets(top_k, candidate_k):
    with pytest.raises(ValueError):
        make_pipeline().search("question", top_k=top_k, candidate_k=candidate_k)
