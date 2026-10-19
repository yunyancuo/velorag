from __future__ import annotations

import time
from typing import Any, Sequence

from velorag.core.config import (
    ChunkingConfig,
    EmbeddingConfig,
    GenerationConfig,
    KnowledgeGraphConfig,
    RerankerConfig,
    RetrievalConfig,
    VeloRAGConfig,
    VectorStoreConfig,
)
from velorag.core.types import (
    ChunkingStrategy,
    EmbeddingBackend,
    GenerationRequest,
    RerankerType,
    VectorStoreBackend,
)
from velorag.eval.dataset import EvalDocument, EvalSample, get_builtin_campus_benchmark
from velorag.eval.metrics import (
    citation_precision,
    keyword_coverage,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from velorag.pipeline import VeloRAG


class EvaluationRunner:
    """End-to-end RAG benchmark runner evaluating Retrieval, Citations, and Refusal Guardrails."""

    def __init__(
        self,
        docs: Sequence[EvalDocument] | None = None,
        samples: Sequence[EvalSample] | None = None,
        top_k: int = 3,
    ):
        builtin_docs, builtin_samples = get_builtin_campus_benchmark()
        self.docs = list(docs) if docs is not None else builtin_docs
        self.samples = list(samples) if samples is not None else builtin_samples
        self.top_k = top_k

    def _build_pipeline(self) -> VeloRAG:
        config = VeloRAGConfig(
            chunking=ChunkingConfig(strategy=ChunkingStrategy.FIXED, chunk_size=260, chunk_overlap=30),
            embedding=EmbeddingConfig(backend=EmbeddingBackend.HASH, dimension=256),
            vector_store=VectorStoreConfig(backend=VectorStoreBackend.IN_MEMORY),
            retrieval=RetrievalConfig(top_k=self.top_k),
            reranker=RerankerConfig(enabled=True, model_type=RerankerType.LEXICAL_OVERLAP),
            knowledge_graph=KnowledgeGraphConfig(enabled=True, max_hops=2),
            generation=GenerationConfig(model_name="offline-extractive", api_key=None),
        )
        rag = VeloRAG(config=config)
        for doc in self.docs:
            rag.ingest_text(
                text=doc.text,
                doc_id=doc.doc_id,
                source=doc.metadata.get("source", doc.title),
                metadata={**doc.metadata, "title": doc.title},
            )
        return rag

    def evaluate_mode(self, mode: str = "graph_enhanced", rag: VeloRAG | None = None) -> dict[str, Any]:
        pipeline = rag or self._build_pipeline()
        use_graph = mode == "graph_enhanced"
        search_mode = "hybrid" if mode == "graph_enhanced" else mode

        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        cit_precisions: list[float] = []
        kw_coverages: list[float] = []
        refusal_correct: list[float] = []
        latencies_ms: list[float] = []
        details: list[dict[str, Any]] = []

        for sample in self.samples:
            t0 = time.perf_counter()
            results = pipeline.search(
                query=sample.query,
                top_k=self.top_k,
                mode=search_mode,
                use_graph=use_graph,
            )
            gen_req = GenerationRequest(
                query=sample.query,
                context_chunks=[r.chunk for r in results],
                min_relevance_score=0.22,
                strict_citation=True,
            )
            gen_resp = pipeline.generator.generate(gen_req, results)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            latencies_ms.append(elapsed_ms)

            retrieved_docs = [r.chunk.doc_id for r in results]
            rel_set = set(sample.relevant_doc_ids)

            if not sample.should_refuse:
                rec = recall_at_k(retrieved_docs, rel_set, k=self.top_k)
                mrr = mrr_at_k(retrieved_docs, rel_set, k=self.top_k)
                ndcg = ndcg_at_k(retrieved_docs, rel_set, k=self.top_k)
                cited_docs = [c.doc_id for c in gen_resp.citations]
                c_prec = citation_precision(cited_docs, rel_set)
                kw_cov = keyword_coverage(gen_resp.content, sample.expected_keywords)

                recalls.append(rec)
                mrrs.append(mrr)
                ndcgs.append(ndcg)
                cit_precisions.append(c_prec)
                kw_coverages.append(kw_cov)
            else:
                rec = 1.0
                mrr = 1.0
                ndcg = 1.0
                c_prec = 1.0 if not gen_resp.citations else 0.0
                kw_cov = 1.0

            is_refusal_correct = 1.0 if (gen_resp.refused == sample.should_refuse) else 0.0
            refusal_correct.append(is_refusal_correct)

            details.append({
                "id": sample.id,
                "category": sample.category,
                "query": sample.query,
                "should_refuse": sample.should_refuse,
                "refused": gen_resp.refused,
                "refusal_correct": bool(is_refusal_correct),
                "recall_at_k": round(rec, 4),
                "mrr_at_k": round(mrr, 4),
                "ndcg_at_k": round(ndcg, 4),
                "citation_precision": round(c_prec, 4),
                "keyword_coverage": round(kw_cov, 4),
                "latency_ms": elapsed_ms,
                "answer_preview": gen_resp.content[:140],
                "citations_count": len(gen_resp.citations),
            })

        def _avg(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        return {
            "mode": mode,
            "top_k": self.top_k,
            "total_samples": len(self.samples),
            "answerable_samples": sum(1 for s in self.samples if not s.should_refuse),
            "unanswerable_samples": sum(1 for s in self.samples if s.should_refuse),
            "metrics": {
                "recall_at_k": _avg(recalls),
                "mrr_at_k": _avg(mrrs),
                "ndcg_at_k": _avg(ndcgs),
                "citation_precision": _avg(cit_precisions),
                "keyword_coverage": _avg(kw_coverages),
                "refusal_accuracy": _avg(refusal_correct),
                "avg_latency_ms": round(sum(latencies_ms) / max(len(latencies_ms), 1), 2),
            },
            "details": details,
        }

    def run_suite(
        self,
        modes: Sequence[str] = ("dense", "sparse", "hybrid", "graph_enhanced"),
    ) -> dict[str, Any]:
        pipeline = self._build_pipeline()
        mode_reports = [self.evaluate_mode(m, rag=pipeline) for m in modes]
        return {
            "benchmark_name": "VeloRAG Campus & AI Lab Grounded QA Benchmark",
            "document_count": len(self.docs),
            "sample_count": len(self.samples),
            "top_k": self.top_k,
            "modes": mode_reports,
            "markdown_table": self.format_markdown_report(mode_reports),
        }

    @staticmethod
    def format_markdown_report(mode_reports: Sequence[dict[str, Any]]) -> str:
        lines = [
            "| Mode | Recall@3 | MRR@3 | NDCG@3 | Citation Precision | Keyword Hit | Refusal Acc | Avg Latency (ms) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for rep in mode_reports:
            m = rep["metrics"]
            lines.append(
                f"| `{rep['mode']}` | {m['recall_at_k']:.2%} | {m['mrr_at_k']:.2%} | "
                f"{m['ndcg_at_k']:.2%} | {m['citation_precision']:.2%} | "
                f"{m['keyword_coverage']:.2%} | {m['refusal_accuracy']:.2%} | {m['avg_latency_ms']:.2f} |"
            )
        return "\n".join(lines)
