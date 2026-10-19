"""Fixed first-40 validation-query challenger; no tuning or fallback models.

Reads the frozen residual pilot, reuses its checked semantic vectors, and scores
the union of dense and hybrid candidates once per query with a real ONNX cross
encoder. Each method still receives only its own original 50 candidates.
"""

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from benchmark_residual import (
    BM25Scorer, Chunk, ResidualConfig, ResidualIndex, embed_data, file_hash,
    load_data, ndcg_at_k, query_rankings, recall_at_k,
)

MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def main(args):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    selected = pilot["development_query_ids"] + pilot["validation_query_ids"]
    corpus, queries, qrels, dev, validation, _ = load_data(
        args.dataset, len(selected), len(pilot["development_query_ids"]), pilot["seed"])
    if dev + validation != selected:
        raise ValueError("Pilot query split does not match input data")
    # Fixed prefix specified before cross-encoder outputs are examined.
    qids = validation[:args.queries]
    if not 0 < args.queries <= len(validation):
        raise ValueError("Require a nonempty subset of pilot validation queries")
    vectors, qvectors, embedding = embed_data(args.dataset, corpus, queries, selected, args.cache, args.threads)
    if embedding["cache_sha256"] != pilot["embedding"]["cache_sha256"]:
        raise ValueError("Semantic vectors differ from pilot")
    qvectors = dict(zip(selected, qvectors))
    chunks = [Chunk(id=str(d["_id"]), doc_id=str(d["_id"]),
                    text=d.get("title", "") + "\n" + d["text"]) for d in corpus]
    index = ResidualIndex(ResidualConfig(**pilot["effective_residual_config"]))
    index.fit(chunks, vectors)
    bm25 = BM25Scorer()
    bm25.index(chunks)
    model = TextCrossEncoder(MODEL, cache_dir=str(args.model_cache), threads=args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, qid in enumerate(qids):
        rankings, _ = query_rankings(queries[qid], qvectors[qid], vectors, index, chunks,
                                     bm25, pilot["selected_weight"], pilot["candidate_k_per_source"])
        sources = {k: rankings[k] for k in ("dense", "hybrid", "residual_hybrid")}
        union = {r.chunk.id: r.chunk for results in sources.values() for r in results}
        started = time.perf_counter()
        scores = list(model.rerank(queries[qid], [c.text for c in union.values()], batch_size=16))
        elapsed = time.perf_counter() - started
        if len(scores) != len(union) or not np.isfinite(scores).all():
            raise ValueError("Invalid cross-encoder predictions; no lexical fallback")
        score_map = dict(zip(union, map(float, scores)))
        modes = {}
        for name, results in sources.items():
            orders = {
                name: results,
                name + "_cross_encoder": sorted(results, key=lambda r: -score_map[r.chunk.id]),
            }
            for mode, ordered in orders.items():
                ids = [r.chunk.doc_id for r in ordered]
                modes[mode] = {
                    "recall_at_10": recall_at_k(ids, qrels[qid], 10),
                    "ndcg_at_10": ndcg_at_k(ids, qrels[qid], 10),
                    "candidate_recall_at_50": recall_at_k(ids, qrels[qid], 50),
                    "candidate_count": len(ids), "top_ids": ids[:10],
                }
        rows.append({"query_id": qid, "relevant_ids": sorted(qrels[qid]), "modes": modes,
                     "union_candidates": len(union), "union_cross_encoder_seconds": elapsed,
                     "union_cross_encoder_scores": score_map})
        # Resumable diagnostic evidence, not a completed report.
        args.out.with_suffix(".partial.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Cross-encoder {position + 1}/{len(qids)} query={qid} union={len(union)} seconds={elapsed:.2f}", flush=True)
    summary = {mode: {metric: float(np.mean([r["modes"][mode][metric] for r in rows]))
                      for metric in ("recall_at_10", "ndcg_at_10", "candidate_recall_at_50")}
               for mode in rows[0]["modes"]}
    report = {
        "experiment": "Fixed first-40 pilot-validation queries; supplemental challenger, no tuning",
        "code_commit_before_run": args.commit, "script_sha256": file_hash(Path(__file__)),
        "pilot_sha256": file_hash(args.pilot), "query_ids": qids,
        "corpus_size": len(corpus), "candidate_k_per_source": 50, "top_k": 10,
        "selected_residual_weight": pilot["selected_weight"],
        "cross_encoder": {"model": MODEL, "fastembed_version": version("fastembed"),
                          "onnx_assets_sha256": {str(p.relative_to(args.model_cache)): file_hash(p)
                                                 for p in args.model_cache.rglob("*.onnx")},
                          "fallback": False},
        "summary": summary, "per_query": rows,
        "limitations": ["40 query prefix only; compare methods within this report, not against full-120 averages.",
                        "Cross-encoder timing covers the shared union, not standalone method latency.",
                        "English scientific abstracts only; no large-corpus or original-algorithm evidence.",
                        "Model token limit truncates long pairs; relevance labels may be incomplete."],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("data/experiments/residual/scifact-pilot-corrected.json"))
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/scifact"))
    parser.add_argument("--cache", type=Path, default=Path("data/benchmarks/semantic-cache"))
    parser.add_argument("--model-cache", type=Path, default=Path("data/benchmarks/reranker-cache"))
    parser.add_argument("--out", type=Path, default=Path("data/experiments/residual/scifact-cross-encoder.json"))
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--commit", default="working-tree")
    main(parser.parse_args())
