import numpy as np
import pytest

from velorag.core.config import EmbeddingConfig, KnowledgeGraphConfig, VeloRAGConfig
from velorag.core.types import Chunk, EmbeddingBackend
from velorag.pipeline import VeloRAG
from velorag.retrieval.residual import ResidualConfig, ResidualIndex, unit_rows


def fixture_index(weight=0.25):
    vectors = np.array([[1, 0.1, 0], [1, 0, 0.2], [1, 0.3, 0], [0, 0, 1]], dtype=np.float32)
    chunks = [Chunk(id=f"c{i}", text=f"evidence {i}", doc_id=f"d{i}",
                    metadata={"kb_id": "a" if i % 2 == 0 else "b"}) for i in range(4)]
    index = ResidualIndex(ResidualConfig(neighbors=2, anchors=2, weight=weight))
    index.fit(chunks, vectors)
    return index, chunks, vectors


def test_zero_residual_weight_matches_cosine_exactly():
    index, chunks, vectors = fixture_index(weight=0)
    query = np.array([1, 0.2, 0])
    expected = unit_rows(vectors) @ unit_rows(query)
    np.testing.assert_allclose(index.scores(query), expected, atol=1e-6)


def test_one_inner_product_equals_original_plus_residual_matching():
    index, _, _ = fixture_index()
    qkey = index.query_key([1, 0.2, 0])
    expected = index._keys[:, :3] @ qkey[:3] + index._keys[:, 3:] @ qkey[3:]
    np.testing.assert_allclose(index.scores([1, 0.2, 0]), expected, atol=1e-6)


def test_filters_apply_before_cutoff_and_missing_filter_returns_empty():
    index, _, _ = fixture_index()
    hits = index.search([1, 0.2, 0], top_k=2, filters={"kb_id": "a"})
    assert len(hits) == 2
    assert {h.chunk.metadata["kb_id"] for h in hits} == {"a"}
    assert index.search([1, 0.2, 0], filters={"kb_id": "missing"}) == []


def test_reordering_inputs_does_not_change_scores():
    index, chunks, vectors = fixture_index()
    other = ResidualIndex(index.config)
    other.fit(chunks[::-1], vectors[::-1])
    np.testing.assert_allclose(index.scores([1, 0.2, 0]), other.scores([1, 0.2, 0]))


def test_singleton_duplicates_and_zero_vectors_are_finite():
    index = ResidualIndex()
    index.fit([Chunk(id="single")], [[1, 0]])
    assert np.isfinite(index.scores([1, 0])).all()
    index.fit([Chunk(id=str(i)) for i in range(3)], [[0, 0], [1, 0], [1, 0]])
    assert np.isfinite(index.scores([0, 0])).all()
    assert np.isfinite(index.scores([1, 0])).all()


def test_add_upsert_delete_and_full_rebuild_are_consistent():
    index, _, _ = fixture_index()
    index.add([Chunk(id="c0", doc_id="replacement")], [[0, 1, 0]])
    assert index.count() == 4
    assert index.search([0, 1, 0])[0].chunk.doc_id == "replacement"
    index.delete(["c0", "c1", "c2", "c3"])
    assert index.count() == 0 and index.search([1, 0, 0]) == []


@pytest.mark.parametrize("vectors", [[[1, 2]], [[1, 2], [float('nan'), 0]]])
def test_invalid_embeddings_fail_visible(vectors):
    with pytest.raises(ValueError):
        ResidualIndex().fit([Chunk(id="a"), Chunk(id="b")], vectors)


def test_pipeline_modes_are_opt_in_and_ingestion_invalidates_index():
    rag = VeloRAG(config=VeloRAGConfig(
        embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=64),
        knowledge_graph=KnowledgeGraphConfig(enabled=False),
    ))
    rag.ingest_text("VeloRAG supports multilingual retrieval.", doc_id="a", metadata={"kb_id": "a"})
    rag.ingest_text("Slurm schedules GPU jobs.", doc_id="b", metadata={"kb_id": "b"})
    assert rag._residual_retriever is None
    baseline = rag.search("VeloRAG retrieval", mode="hybrid", use_graph=False)
    assert baseline and rag._residual_retriever is None
    for mode in ("local_residual", "residual_hybrid"):
        hits = rag.search("VeloRAG retrieval", mode=mode, filters={"kb_id": "a"})
        assert hits and all(r.chunk.metadata["kb_id"] == "a" for r in hits)
    assert rag._residual_retriever is not None
    rag.ingest_text("New retrieval evidence.", doc_id="c")
    assert rag._residual_retriever is None
