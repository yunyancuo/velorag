"""Comprehensive tests for VeloRAG v0.3 features."""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_multilingual_bm25_and_rrf():
    from velorag.core.types import Chunk
    from velorag.retrieval.engine import BM25Scorer, reciprocal_rank_fusion, tokenize_multilingual

    tokens = tokenize_multilingual("VeloRAG支持混合检索与知识图谱")
    assert "velorag" in tokens
    assert "混合" in tokens
    assert "检索" in tokens

    chunks = [
        Chunk(id="c1", text="硕士研究生培养体系要求总学分不少于32学分，其中学位必修课18学分。", doc_id="d1", chunk_index=0),
        Chunk(id="c2", text="AI开源实验室算力集群采用Slurm作业调度系统管理。", doc_id="d2", chunk_index=0),
    ]
    bm25 = BM25Scorer()
    bm25.index(chunks)

    res = bm25.search("硕士研究生总学分要求多少？", top_k=2)
    assert len(res) >= 1
    assert res[0].chunk.id == "c1"

    fused = reciprocal_rank_fusion([res, res], weights=[0.6, 0.4])
    assert len(fused) >= 1
    assert fused[0].chunk.id == "c1"


def test_bilingual_knowledge_graph_and_subgraph():
    from velorag.core.types import Chunk
    from velorag.graph.knowledge_graph import KnowledgeGraph

    kg = KnowledgeGraph()
    chunks = [
        Chunk(
            id="c_kg1",
            text="《研究生培养手册》规定人工智能学院包含医疗AI实验室。VeloRAG支持知识图谱。",
            doc_id="d_kg1",
            chunk_index=0,
        )
    ]
    kg.extract_from_chunks(chunks)
    assert kg.node_count > 0
    assert kg.edge_count > 0

    sub = kg.get_subgraph(query="人工智能学院", limit=10)
    assert "nodes" in sub
    assert "edges" in sub
    assert len(sub["nodes"]) > 0


def test_grounded_citations_and_refusal_guardrail():
    from velorag.server.state import get_default_pipeline

    rag = get_default_pipeline(seed_builtin=True)

    # 1. Answerable question -> grounded answer with [Source N] citation
    ans = rag.query("硕士研究生毕业总学分和学位必修课分别要求不少于多少学分？", top_k=3)
    assert ans.refused is False
    assert "32" in ans.content
    assert len(ans.citations) >= 1
    assert ans.citations[0].index >= 1

    # 2. Unanswerable trap question -> refused is True
    trap = rag.query("请问紫金港校区第三食堂的夜宵麻辣烫几点关门？", top_k=3)
    assert trap.refused is True
    assert trap.citations == []


def test_agent_canvas_dag_and_serialization():
    from velorag.agent import AgentCanvas, CodeComponent
    from velorag.server.state import get_default_pipeline

    canvas = AgentCanvas.create_default_rag_canvas("default")
    code_node = CodeComponent(
        "code_1",
        "Custom Variable Tag",
        code="output = {'custom_flag': True}",
    )
    canvas.add_component(code_node)

    # Round-trip JSON serialization
    payload = canvas.to_dict()
    restored = AgentCanvas.from_dict(payload)
    assert restored.name == canvas.name
    assert len(restored.get_components()) == len(canvas.get_components())

    pipeline = get_default_pipeline(seed_builtin=True)
    ctx = asyncio.run(restored.run("AI开源实验室算力集群使用什么调度系统？", pipeline=pipeline))
    assert "Slurm" in ctx.final_answer
    assert ctx.refused is False
    assert len(ctx.trace) >= 4
    assert ctx.variables.get("custom_flag") is True


def test_evaluation_runner_suite():
    from velorag.eval.runner import EvaluationRunner

    runner = EvaluationRunner(top_k=3)
    suite = runner.run_suite(modes=["dense", "sparse", "hybrid", "graph_enhanced"])
    assert suite["document_count"] == 5
    assert suite["sample_count"] == 8
    assert len(suite["modes"]) == 4
    assert "| Mode |" in suite["markdown_table"]

    for mode_rep in suite["modes"]:
        metrics = mode_rep["metrics"]
        assert metrics["recall_at_k"] >= 0.8
        assert metrics["refusal_accuracy"] >= 0.85


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from velorag.server.app import app

    with TestClient(app) as client:
        r_health = client.get("/health")
        assert r_health.status_code == 200
        assert r_health.json()["status"] == "ok"

        r_kbs = client.get("/api/v1/knowledge")
        assert r_kbs.status_code == 200
        assert len(r_kbs.json()) >= 1

        r_query = client.post(
            "/api/v1/query",
            json={"query": "医疗AI联合培养项目中，医学影像数据在训练前必须完成什么处理？", "top_k": 3},
        )
        assert r_query.status_code == 200
        q_data = r_query.json()
        assert q_data["refused"] is False
        assert "DICOM" in q_data["answer"]
        assert len(q_data["citations"]) >= 1

        r_canvases = client.get("/api/v1/agents")
        assert r_canvases.status_code == 200
        canvas_id = r_canvases.json()[0]["id"]

        r_run = client.post(
            f"/api/v1/agents/{canvas_id}/run",
            json={"query": "参加CCF-A类国际会议最高可申请多少差旅资助？"},
        )
        assert r_run.status_code == 200
        assert "15000" in r_run.json()["final_answer"]

        r_eval = client.post("/api/v1/eval/run", json={"modes": ["hybrid", "graph_enhanced"], "top_k": 3})
        assert r_eval.status_code == 200
        assert len(r_eval.json()["modes"]) == 2
