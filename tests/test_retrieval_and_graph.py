"""四路检索过滤、向量分片、知识图谱与 Pipeline 缓存深度回归测试。"""

from __future__ import annotations

from velorag.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    KnowledgeGraphConfig,
    RerankerConfig,
    VectorStoreConfig,
    VeloRAGConfig,
)
from velorag.core.settings import IndexSettings
from velorag.core.types import (
    Chunk,
    ChunkingStrategy,
    EmbeddingBackend,
    RerankerType,
    VectorStoreBackend,
)
from velorag.graph.knowledge_graph import KnowledgeGraph
from velorag.indexing.sharding import ShardedIndex
from velorag.indexing.vector_store import InMemoryVectorStore
from velorag.pipeline import VeloRAG


def _make_rag() -> VeloRAG:
    cfg = VeloRAGConfig(
        chunking=ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=260, chunk_overlap=30),
        embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=128),
        vector_store=VectorStoreConfig(backend=VectorStoreBackend.IN_MEMORY),
        reranker=RerankerConfig(enabled=True, model_type=RerankerType.LEXICAL_OVERLAP),
        knowledge_graph=KnowledgeGraphConfig(enabled=True, max_hops=2),
        generation=GenerationConfig(
            model_name="custom-test-model",
            api_base="https://example.internal/v1",
            api_key="",
        ),
    )
    return VeloRAG(config=cfg)


def test_all_retrieval_modes_respect_kb_and_metadata_filters():
    rag = _make_rag()
    rag.ingest_text(
        text="AlphaLab uses OrionScheduler for cluster orchestration. OrionScheduler supports GPU jobs.",
        doc_id="doc_alpha",
        source="alpha.md",
        metadata={"kb_id": "kb_alpha", "department": "ai"},
    )
    rag.ingest_text(
        text="BetaLab uses OrionScheduler for biology simulation. OrionScheduler runs genomics jobs.",
        doc_id="doc_beta",
        source="beta.md",
        metadata={"kb_id": "kb_beta", "department": "bio"},
    )

    for mode in ("dense", "sparse", "hybrid", "graph_enhanced"):
        res_alpha = rag.search(
            "OrionScheduler",
            top_k=5,
            mode=mode,
            use_graph=(mode == "graph_enhanced"),
            filters={"kb_id": "kb_alpha"},
        )
        assert len(res_alpha) >= 1
        assert all(r.chunk.metadata.get("kb_id") == "kb_alpha" for r in res_alpha)
        assert all(r.chunk.doc_id == "doc_alpha" for r in res_alpha)

        # 多字段过滤 + $nin 排除过滤
        res_bio = rag.search(
            "OrionScheduler",
            top_k=5,
            mode=mode,
            use_graph=(mode == "graph_enhanced"),
            filters={"kb_id": {"$nin": ["kb_alpha"]}, "department": "bio"},
        )
        assert len(res_bio) >= 1
        assert all(r.chunk.metadata.get("kb_id") == "kb_beta" for r in res_bio)


def test_dense_filter_after_topk_regression():
    """验证即使其他知识库有大量更高分分块，按 kb_id 过滤仍能完整召回目标知识库。"""
    store = InMemoryVectorStore(dimension=4)
    query_vec = [1.0, 0.0, 0.0, 0.0]

    # 插入 15 条在 kb_noise 中相似度为 1.0 的分块
    noise_chunks = [
        Chunk(id=f"noise_{i}", text=f"noise {i}", doc_id="d_noise", metadata={"kb_id": "kb_noise"})
        for i in range(15)
    ]
    store.add(noise_chunks, [[1.0, 0.0, 0.0, 0.0] for _ in noise_chunks])

    # 插入 2 条在 kb_target 中相似度略低的分块
    target_chunks = [
        Chunk(id="target_1", text="target one", doc_id="d_t1", metadata={"kb_id": "kb_target"}),
        Chunk(id="target_2", text="target two", doc_id="d_t2", metadata={"kb_id": "kb_target"}),
    ]
    store.add(target_chunks, [[0.8, 0.6, 0.0, 0.0], [0.7, 0.7, 0.0, 0.0]])

    res = store.search(query_vec, top_k=3, filters={"kb_id": "kb_target"})
    assert [r.chunk.id for r in res] == ["target_1", "target_2"]


def test_vector_store_idempotent_upsert_and_boundaries():
    store = InMemoryVectorStore(dimension=3)
    c1 = Chunk(id="dup_1", text="v1", doc_id="d1", metadata={"kb_id": "k1"})
    store.add([c1], [[1.0, 0.0, 0.0]])
    assert store.count() == 1

    # 同 ID 重复写入应覆盖更新而不是产生重复项
    c1_v2 = Chunk(id="dup_1", text="v2", doc_id="d1", metadata={"kb_id": "k1"})
    store.add([c1_v2], [[0.0, 1.0, 0.0]])
    assert store.count() == 1

    hits = store.search([0.0, 1.0, 0.0], top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk.text == "v2"

    # 零向量查询与 top_k <= 0 应安全返回空列表
    assert store.search([0.0, 0.0, 0.0], top_k=5) == []
    assert store.search([0.0, 1.0, 0.0], top_k=0) == []


def test_sharded_index_filters_and_delete():
    settings = IndexSettings(num_shards=4)
    index = ShardedIndex(settings)
    chunks = [
        Chunk(id=f"c_{i}", text=f"chunk {i}", doc_id=f"d_{i}", metadata={"kb_id": "k_a" if i % 2 == 0 else "k_b"})
        for i in range(12)
    ]
    embeddings = [[1.0, float(i) * 0.01, 0.0, 0.0] for i in range(12)]
    index.add(chunks, embeddings)
    assert index.count() == 12

    res_a = index.search([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"kb_id": "k_a"})
    assert len(res_a) == 5
    assert all(r.chunk.metadata["kb_id"] == "k_a" for r in res_a)

    index.delete(["c_0", "c_2", "c_4"])
    assert index.count() == 9


def test_knowledge_graph_empty_query_multihop_cycles_and_cleanup():
    kg = KnowledgeGraph()
    chunks = [
        Chunk(id="c1", text="AlphaSys depends on BetaEngine.", doc_id="d1"),
        Chunk(id="c2", text="BetaEngine depends on GammaCore.", doc_id="d2"),
        Chunk(id="c3", text="GammaCore depends on AlphaSys.", doc_id="d3"),
        Chunk(id="c_priv", text="PrivateVault contains SecretKey.", doc_id="d4"),
    ]
    kg.extract_from_chunks(chunks)

    # 空查询不得误返回全部实体
    assert kg.find_relevant_entities("") == []
    assert kg.find_relevant_entities("   ") == []

    # 环路图多跳遍历不应死循环，且 max_hops=2 能触及 c1/c2/c3
    alpha_entities = kg.find_relevant_entities("AlphaSys")
    assert len(alpha_entities) >= 1
    dist_map = kg.get_chunks_with_distance(alpha_entities[0].id, max_distance=2)
    assert "c1" in dist_map
    assert "c2" in dist_map

    # 严格遵守 limit 与 allowed_chunk_ids 租户隔离
    sub_limited = kg.get_subgraph(query="AlphaSys", max_hops=2, limit=2)
    assert len(sub_limited["nodes"]) <= 2

    sub_isolated = kg.get_subgraph(query=None, limit=50, allowed_chunk_ids={"c1", "c2", "c3"})
    isolated_names = {n["name"] for n in sub_isolated["nodes"]}
    assert "PrivateVault" not in isolated_names
    assert "SecretKey" not in isolated_names

    # 删除 chunk 应级联清理孤立边与孤立实体
    kg.remove_chunks({"c_priv"})
    assert kg.find_relevant_entities("PrivateVault") == []


def test_pipeline_cache_query_filters_and_config_propagation():
    rag = _make_rag()
    assert rag.generator.model == "custom-test-model"
    assert rag.generator.api_base == "https://example.internal/v1"

    rag.ingest_text(
        text="Finance policy states reimbursement deadline is 15 working days.",
        doc_id="d_fin",
        source="finance.md",
        metadata={"kb_id": "kb_fin", "dept": "finance"},
    )
    rag.ingest_text(
        text="Engineering guide recommends Pytest with coverage checks.",
        doc_id="d_eng",
        source="eng.md",
        metadata={"kb_id": "kb_eng", "dept": "engineering"},
    )

    # 缓存键对 filters 键顺序不敏感，且外部修改返回值不污染缓存
    r1 = rag.search("reimbursement deadline", top_k=3, filters={"kb_id": "kb_fin", "dept": "finance"})
    assert len(r1) == 1
    original_score = r1[0].score
    r1[0].score = -999.0

    r2 = rag.search("reimbursement deadline", top_k=3, filters={"dept": "finance", "kb_id": "kb_fin"})
    assert len(r2) == 1
    assert r2[0].score == original_score

    # 空结果也应正确命中缓存
    empty1 = rag.search("nonexistent_kb_query", top_k=3, filters={"kb_id": "kb_missing"})
    assert empty1 == []
    assert rag.cache.size() >= 2

    # VeloRAG.query 支持 filters 隔离：在 kb_eng 中查财务问题应拒答
    ans_eng = rag.query("What is the reimbursement deadline?", top_k=3, filters={"kb_id": "kb_eng"})
    assert ans_eng.refused is True

    ans_fin = rag.query("What is the reimbursement deadline?", top_k=3, filters={"kb_id": "kb_fin"})
    assert ans_fin.refused is False
    assert "15" in ans_fin.content
