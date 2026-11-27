"""文档解析分块、Agent Canvas 拓扑与环路检测、Eval 指标与 LRU 缓存测试。"""

from __future__ import annotations

import asyncio
import io
import math
import threading
import time
from pathlib import Path

import pytest
from docx import Document as DocxDocument

from velorag.agent import (
    AgentCanvas,
    BeginComponent,
    CodeComponent,
    EndComponent,
    RetrievalComponent,
)
from velorag.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    KnowledgeGraphConfig,
    VectorStoreConfig,
    VeloRAGConfig,
)
from velorag.core.settings import ChunkSettings
from velorag.core.types import (
    ChunkingStrategy,
    ChunkStrategy,
    Document,
    EmbeddingBackend,
    VectorStoreBackend,
)
from velorag.document.chunker import Chunker
from velorag.document.loader import DocumentLoader
from velorag.document.parser import DocxParser, MarkdownParser, PDFParser
from velorag.eval.metrics import (
    citation_precision,
    keyword_coverage,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from velorag.eval.runner import EvaluationRunner
from velorag.pipeline import VeloRAG
from velorag.utils.cache import MemoryCache


def test_chunker_recursive_no_duplication_and_edge_cases():
    # 回归测试：小段 + 超大段 + 小段在旧实现中会因未重置 current 导致重复输出第一段
    text = "Intro alpha.\n\n" + ("LongSentencePart " * 12) + "\n\nOutro omega."
    doc = Document(id="doc_rec", content=text, source="rec.txt")
    chunker = Chunker(ChunkSettings(strategy=ChunkStrategy.RECURSIVE, chunk_size=60, chunk_overlap=10))
    chunks = chunker.chunk(doc)

    assert len(chunks) >= 3
    intro_hits = sum(1 for c in chunks if "Intro alpha." in c.text)
    outro_hits = sum(1 for c in chunks if "Outro omega." in c.text)
    assert intro_hits == 1
    assert outro_hits == 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # 空白文本和极端参数安全处理
    empty_doc = Document(id="empty", content="   \n\t  ")
    for strat in (
        ChunkStrategy.FIXED,
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.SENTENCE,
        ChunkStrategy.SEMANTIC,
    ):
        c = Chunker(ChunkSettings(strategy=strat, chunk_size=10, chunk_overlap=999))
        assert c.chunk(empty_doc) == []
        res = c.chunk(Document(id="s", content="Hello world! Another sentence here."))
        assert len(res) >= 1


def test_document_loader_recursive_size_limit_and_parsers(tmp_path: Path):
    top_file = tmp_path / "top.md"
    top_file.write_text("# Title\nSee [VeloRAG](https://example.com) ![img](a.png)", encoding="utf-8")
    sub_dir = tmp_path / "nested"
    sub_dir.mkdir()
    sub_file = sub_dir / "nested.txt"
    sub_file.write_text("Nested document text.", encoding="utf-8")

    loader = DocumentLoader()
    non_rec = loader.load(tmp_path, recursive=False)
    rec = loader.load(tmp_path, recursive=True)
    assert len(non_rec) == 1
    assert len(rec) == 2

    # MarkdownParser 应去除图片并保留链接文字
    md_doc = MarkdownParser().parse(top_file.read_bytes())
    assert "VeloRAG" in md_doc.content
    assert "![img]" not in md_doc.content

    # 文件大小上限限制
    tiny_loader = DocumentLoader(max_file_size_bytes=10)
    with pytest.raises(ValueError, match="超过上限"):
        tiny_loader.load(top_file)

    # DocxParser 同时提取段落与表格文本
    buf = io.BytesIO()
    docx = DocxDocument()
    docx.add_paragraph("Paragraph summary content")
    table = docx.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Recall@3"
    docx.save(buf)
    parsed_docx = DocxParser().parse(buf.getvalue())
    assert "Paragraph summary content" in parsed_docx.content
    assert "Metric | Recall@3" in parsed_docx.content

    # 损坏 PDF/DOCX 抛出清晰 ValueError
    with pytest.raises(ValueError, match="PDF"):
        PDFParser().parse(b"not a pdf")
    with pytest.raises(ValueError, match="DOCX"):
        DocxParser().parse(b"not a docx")


def test_agent_canvas_cycle_detection_orphan_and_kb_scope():
    # 1. 自环直接拒绝
    c_self = AgentCanvas("SelfLoop")
    with pytest.raises(ValueError, match="自环"):
        c_self.add_edge("begin", "begin")

    # 2. 多跳环路检测
    c_cycle = AgentCanvas("CycleCanvas")
    n1 = CodeComponent("n1", "N1", code="output = {'a': 1}")
    n2 = CodeComponent("n2", "N2", code="output = {'b': 2}")
    c_cycle.add_component(n1)
    c_cycle.add_component(n2)
    c_cycle.add_edge("n2", "n1")
    report = c_cycle.validate()
    assert report["valid"] is False
    assert any("环路" in err for err in report["errors"])
    with pytest.raises(ValueError, match="环路"):
        asyncio.run(c_cycle.run("loop test"))

    # 3. Switch 分支环路检测
    c_sw = AgentCanvas.from_dict(
        {
            "name": "SwitchCycle",
            "nodes": [
                {"id": "begin", "type": "begin", "name": "Begin"},
                {
                    "id": "sw1",
                    "type": "switch",
                    "name": "Sw",
                    "config": {"true_target": "begin", "false_target": "end1"},
                },
                {"id": "end1", "type": "end", "name": "End"},
            ],
            "edges": [{"source": "begin", "target": "sw1", "label": "default"}],
        }
    )
    assert c_sw.validate()["valid"] is False

    # 4. 孤立节点诊断 & 缺失 Begin 节点拒绝
    c_orphan = AgentCanvas.from_dict(
        {
            "name": "OrphanCanvas",
            "nodes": [
                {"id": "begin", "type": "begin", "name": "Begin"},
                {"id": "end1", "type": "end", "name": "End"},
                {"id": "orphan1", "type": "end", "name": "Orphan"},
            ],
            "edges": [{"source": "begin", "target": "end1", "label": "default"}],
        }
    )
    rep_orphan = c_orphan.validate()
    assert rep_orphan["valid"] is True
    assert "orphan1" in rep_orphan["unreachable_nodes"]

    c_no_begin = AgentCanvas.from_dict(
        {"name": "NoBegin", "nodes": [{"id": "end1", "type": "end", "name": "End"}], "edges": []}
    )
    assert c_no_begin.validate()["valid"] is False

    # 5. 节点执行失败错误记录与传播
    c_err = AgentCanvas("ErrorCanvas")

    def _boom(_ctx):
        raise ZeroDivisionError("boom")

    c_err.add_component(CodeComponent("bad", "Bad", fn=_boom))
    with pytest.raises(RuntimeError, match="bad"):
        asyncio.run(c_err.run("test"))

    # 6. RetrievalComponent 严格传递 kb_id 隔离过滤
    rag = VeloRAG(
        config=VeloRAGConfig(
            chunking=ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=200, chunk_overlap=20),
            embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=128),
            vector_store=VectorStoreConfig(backend=VectorStoreBackend.IN_MEMORY),
            knowledge_graph=KnowledgeGraphConfig(enabled=True, max_hops=2),
        )
    )
    rag.ingest_text("Canvas KB1 contains ApolloEngine.", doc_id="d1", metadata={"kb_id": "kb_1"})
    rag.ingest_text("Canvas KB2 contains ZeusEngine.", doc_id="d2", metadata={"kb_id": "kb_2"})

    c_kb = AgentCanvas("ScopedCanvas")
    c_kb.add_component(BeginComponent("begin"))
    c_kb.add_component(RetrievalComponent("ret", "Ret", kb_id="kb_1", top_k=5, mode="graph_enhanced"))
    c_kb.add_component(EndComponent("end"))
    ctx = asyncio.run(c_kb.run("ZeusEngine", pipeline=rag))
    chunks = ctx.variables["retrieved_chunks"]
    assert all(item["metadata"].get("kb_id") == "kb_1" for item in chunks)


def test_eval_metrics_math_and_dynamic_report_header():
    assert recall_at_k(["d1", "d1", "d2"], {"d1", "d3"}, k=2) == pytest.approx(0.5)
    assert recall_at_k(["d1"], {"d1"}, k=0) == 0.0
    assert recall_at_k([], {"d1"}, k=3) == 0.0

    assert mrr_at_k(["d0", "d2", "d1"], {"d1"}, k=3) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(["d0", "d2"], {"d1"}, k=2) == 0.0
    assert mrr_at_k(["d1"], {"d1"}, k=0) == 0.0

    expected_dcg = 1.0 / math.log2(3)  # 第 2 位命中 (idx=1 -> log2(3))
    expected_idcg = 1.0 / math.log2(2)
    assert ndcg_at_k(["d0", "d1"], {"d1"}, k=2) == pytest.approx(expected_dcg / expected_idcg)
    assert ndcg_at_k(["d1"], {"d1"}, k=0) == 0.0

    assert citation_precision([], set()) == 1.0
    assert citation_precision([], {"d1"}) == 0.0
    assert citation_precision(["d1", "d2"], {"d1"}) == pytest.approx(0.5)
    assert keyword_coverage("VeloRAG uses RRF", ["velorag", "rrf", "missing"]) == pytest.approx(2 / 3)
    assert keyword_coverage("", ["velorag"]) == 0.0

    runner = EvaluationRunner(top_k=5)
    suite = runner.run_suite(modes=["hybrid"])
    assert "| Mode | Recall@5 | MRR@5 | NDCG@5 |" in suite["markdown_table"]


def test_memory_cache_lru_ttl_and_thread_safety():
    # LRU 淘汰验证
    cache = MemoryCache(max_entries=2)
    cache.set("k1", 1)
    cache.set("k2", 2)
    assert cache.get("k1") == 1  # 访问 k1，使 k2 成为最久未使用
    cache.set("k3", 3)
    assert cache.get("k1") == 1
    assert cache.get("k2") is None
    assert cache.get("k3") == 3

    # 更新已有键不应误删其他键
    cache.set("k1", 10)
    assert cache.size() == 2
    assert cache.get("k1") == 10
    assert cache.get("k3") == 3

    # TTL 过期验证
    ttl_cache = MemoryCache(max_entries=10, ttl_seconds=0.05)
    ttl_cache.set("temp", "alive")
    assert ttl_cache.get("temp") == "alive"
    time.sleep(0.08)
    assert ttl_cache.get("temp") is None
    assert ttl_cache.size() == 0

    # max_entries=0 禁用缓存不崩溃
    disabled = MemoryCache(max_entries=0)
    disabled.set("a", 1)
    assert disabled.get("a") is None
    assert disabled.size() == 0

    # 多线程并发压力测试
    conc_cache = MemoryCache(max_entries=100)
    errors: list[Exception] = []

    def _worker(tid: int) -> None:
        try:
            for i in range(150):
                key = f"t{tid}_{i % 40}"
                conc_cache.set(key, i)
                _ = conc_cache.get(key)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert conc_cache.size() <= 100
