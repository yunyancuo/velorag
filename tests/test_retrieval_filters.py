import pytest

from velorag.core.config import EmbeddingConfig, VeloRAGConfig
from velorag.core.types import EmbeddingBackend
from velorag.pipeline import VeloRAG


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", "graph_enhanced"])
@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"kb_id": "alpha"}, {"alpha"}),
        ({"doc_id": "beta"}, {"beta"}),
        ({"kb_id": "alpha", "doc_id": "beta"}, set()),
        ({"kb_id": "missing"}, set()),
    ],
)
def test_retrieval_respects_filters_across_sources(mode, filters, expected):
    rag = VeloRAG(config=VeloRAGConfig(
        embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=256),
    ))
    # Identical content forces both KBs into the same graph entity neighborhood.
    for kb_id in ("alpha", "beta"):
        rag.ingest_text(
            "VeloRAG supports Knowledge Graph retrieval.",
            doc_id=kb_id,
            source=f"{kb_id}.md",
            metadata={"kb_id": kb_id},
        )

    results = rag.search("VeloRAG Knowledge Graph", top_k=10, mode=mode, filters=filters)
    assert {result.chunk.doc_id for result in results} == expected
