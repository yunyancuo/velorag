"""Bounded semantic retrieval pilot on SciFact train qrels, never official test.

Uses FastEmbed BGE with explicit failure instead of a HASH fallback. Selects the
residual weight only on development queries, then evaluates untouched validation
queries from the training split. This is not a large-corpus scalability test.
"""

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from velorag.core.types import Chunk, SearchResult
from velorag.eval.metrics import ndcg_at_k, recall_at_k
from velorag.retrieval.engine import BM25Scorer, min_max_normalize, reciprocal_rank_fusion
from velorag.retrieval.residual import ResidualConfig, ResidualIndex, unit_rows
from velorag.retrieval.reranker import LexicalOverlapReranker


MODEL = "BAAI/bge-small-en-v1.5"
WEIGHTS = (0.0, 0.1, 0.25, 0.5)


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def load_data(folder, query_count, dev_count, seed):
    corpus = [json.loads(line) for line in (folder / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus.sort(key=lambda doc: str(doc["_id"]))
    ids = {str(doc["_id"]) for doc in corpus}
    queries = {str(q["_id"]): q["text"] for q in map(json.loads, (folder / "queries.jsonl").read_text(encoding="utf-8").splitlines())}
    qrels = {}
    with (folder / "qrels/train.tsv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    if any(not relevant <= ids for relevant in qrels.values()):
        raise ValueError("Relevant documents are missing from the corpus")
    if not set(qrels) <= set(queries):
        raise ValueError("Judged queries are missing from the query file")
    selected = sorted(qrels)
    random.Random(seed).shuffle(selected)
    if not 0 < dev_count < query_count <= len(selected):
        raise ValueError("Require 0 < dev_count < query_count <= judged training queries")
    return corpus, queries, qrels, selected[:dev_count], selected[dev_count:query_count]


def embed_data(folder, corpus, queries, selected, cache_dir, threads):
    from fastembed import TextEmbedding

    texts = [doc.get("title", "") + "\n" + doc["text"] for doc in corpus]
    fingerprint = sha256(json.dumps({
        "model": MODEL, "fastembed": version("fastembed"),
        "corpus_sha256": file_hash(folder / "corpus.jsonl"),
        "query_texts": [(qid, queries[qid]) for qid in selected],
        "encoding": "title-newline-text; passage_embed/query_embed",
    }, sort_keys=True).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{fingerprint}.npz"
    meta_path = cache_dir / f"{fingerprint}.json"
    if cache_path.exists():
        provenance = json.loads(meta_path.read_text(encoding="utf-8"))
        if file_hash(cache_path) != provenance["cache_sha256"]:
            raise ValueError("Cached embeddings differ from their provenance checksum")
        with np.load(cache_path, allow_pickle=False) as cached:
            vectors, query_vectors = cached["documents"], cached["queries"]
        seconds = None
    else:
        model = TextEmbedding(model_name=MODEL, cache_dir=str(cache_dir / "models"), threads=threads)
        started = time.perf_counter()
        print(f"Encoding {len(texts)} documents with {MODEL}", flush=True)
        vectors = np.asarray(list(model.passage_embed(texts, batch_size=64)), dtype=np.float32)
        print(f"Encoding {len(selected)} train-split queries", flush=True)
        query_vectors = np.asarray(list(model.query_embed([queries[qid] for qid in selected], batch_size=64)), dtype=np.float32)
        seconds = time.perf_counter() - started
        np.savez_compressed(cache_path, documents=vectors, queries=query_vectors)
        provenance = {
            "cache_sha256": file_hash(cache_path),
            "onnx_assets_sha256": {str(p.relative_to(cache_dir)): file_hash(p)
                                   for p in (cache_dir / "models").rglob("*.onnx")},
        }
        meta_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if vectors.shape != (len(corpus), 384) or query_vectors.shape != (len(selected), 384):
        raise ValueError("Unexpected BGE embedding shape")
    if not np.isfinite(vectors).all() or not np.isfinite(query_vectors).all():
        raise ValueError("Non-finite semantic embeddings")
    return unit_rows(vectors), unit_rows(query_vectors), {
        "model": MODEL, "fastembed_version": version("fastembed"),
        "onnxruntime_version": version("onnxruntime"),
        "cache_fingerprint": fingerprint, "embedding_seconds_this_run": seconds,
        **provenance,
        "hash_fallback": False,
    }


def top_results(scores, chunks, depth, source):
    order = np.argsort(-scores, kind="stable")[:depth]
    return [SearchResult(chunks[i], float(scores[i]), source) for i in order]


def fuse(dense, sparse, depth):
    copies = [[SearchResult(r.chunk, r.score, r.retrieval_source) for r in results]
              for results in (dense, sparse)]
    return reciprocal_rank_fusion([min_max_normalize(r) for r in copies], weights=[0.5, 0.5])[:depth]


def query_rankings(query, qvector, vectors, index, chunks, bm25, weight, depth):
    dense_scores = vectors @ qvector
    residual_scores = (1 - weight) * dense_scores + weight * index.scores(qvector)
    dense = top_results(dense_scores, chunks, depth, "dense")
    residual = top_results(residual_scores, chunks, depth, "local_residual")
    sparse = bm25.search(query, top_k=depth)
    hybrid, residual_hybrid = fuse(dense, sparse, depth), fuse(residual, sparse, depth)
    reranker = LexicalOverlapReranker()
    return {
        "bm25": sparse, "dense": dense, "local_residual": residual,
        "hybrid": hybrid, "residual_hybrid": residual_hybrid,
        "hybrid_lexical_rerank": reranker.rerank(query, hybrid),
        "residual_hybrid_lexical_rerank": reranker.rerank(query, residual_hybrid),
    }


def summarize(rows, k, depth):
    modes = rows[0]["modes"]
    return {mode: {
        f"recall_at_{k}": float(np.mean([r["modes"][mode]["recall"] for r in rows])),
        f"ndcg_at_{k}": float(np.mean([r["modes"][mode]["ndcg"] for r in rows])),
        f"candidate_recall_at_{depth}": float(np.mean([r["modes"][mode]["candidate_recall"] for r in rows])),
    } for mode in modes}


def main(args):
    corpus, queries, qrels, dev, validation = load_data(args.dataset, args.queries, args.dev_queries, args.seed)
    selected = dev + validation
    vectors, qvectors, embedding_meta = embed_data(args.dataset, corpus, queries, selected, args.cache, args.threads)
    query_vectors = dict(zip(selected, qvectors))
    chunks = [Chunk(id=str(d["_id"]), doc_id=str(d["_id"]), text=d.get("title", "") + "\n" + d["text"]) for d in corpus]
    config = ResidualConfig(neighbors=16, anchors=64, weight=1.0, seed=args.seed)
    index = ResidualIndex(config)
    print("Constructing exact one-layer neighborhood references", flush=True)
    index.fit(chunks, vectors)
    if [c.id for c in index.chunks] != [c.id for c in chunks]:
        raise ValueError("Residual and baseline document orders disagree")
    bm25 = BM25Scorer()
    bm25.index(chunks)

    def evaluate(qids, weight):
        rows = []
        for qid in qids:
            ranks = query_rankings(queries[qid], query_vectors[qid], vectors, index, chunks, bm25, weight, args.candidates)
            modes = {}
            for mode, results in ranks.items():
                ids = [r.chunk.doc_id for r in results]
                modes[mode] = {
                    "recall": recall_at_k(ids, qrels[qid], args.top_k),
                    "ndcg": ndcg_at_k(ids, qrels[qid], args.top_k),
                    "candidate_recall": recall_at_k(ids, qrels[qid], args.candidates),
                    "top_ids": ids[:args.top_k],
                }
            rows.append({"query_id": qid, "relevant_ids": sorted(qrels[qid]), "modes": modes})
        return rows

    # Only development labels choose a weight. Ties choose the smaller weight,
    # including zero: the experiment may legitimately reject residual matching.
    tuning = []
    for weight in WEIGHTS:
        print(f"Development weight={weight}", flush=True)
        rows = evaluate(dev, weight)
        metrics = summarize(rows, args.top_k, args.candidates)
        tuning.append({"weight": weight, "metrics": metrics})
    metric_name = f"ndcg_at_{args.top_k}"
    chosen = max(tuning, key=lambda r: (r["metrics"]["residual_hybrid"][metric_name], -r["weight"]))["weight"]
    print(f"Frozen development-selected weight={chosen}; evaluating validation", flush=True)
    rows = evaluate(validation, chosen)
    report = {
        "experiment": "SciFact train-split development pilot, not official test or scale evidence",
        "code_commit_before_run": args.commit,
        "source_sha256": {str(p): file_hash(p) for p in (
            Path(__file__), Path("src/velorag/retrieval/residual.py"),
            Path("src/velorag/retrieval/engine.py"), Path("src/velorag/eval/metrics.py"))},
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "seed": args.seed,
        "dataset_sha256": {str(p.relative_to(args.dataset)): file_hash(p) for p in (
            args.dataset / "corpus.jsonl", args.dataset / "queries.jsonl", args.dataset / "qrels/train.tsv")},
        "corpus_size": len(corpus), "development_query_ids": dev,
        "validation_query_ids": validation, "embedding": embedding_meta,
        "top_k": args.top_k, "candidate_k_per_source": args.candidates,
        "index": index.stats(), "effective_residual_config": {**asdict(config), "weight": chosen},
        "weight_grid": list(WEIGHTS), "selection_metric": f"development residual_hybrid {metric_name}",
        "selected_weight": chosen, "development": tuning,
        "validation": summarize(rows, args.top_k, args.candidates), "per_query_validation": rows,
        "limitations": [
            "Single public dataset; no large-corpus, streaming-update, multilingual, or business-data validation.",
            "Simplified NEAR-inspired algorithm, not a faithful full-paper reproduction or original contribution.",
            "Query-level split of train qrels, not source-document-disjoint; official test qrels unused.",
            "Baselines use a lexical reranker; no learned cross-encoder comparison in this pilot.",
            "Exact search; quadratic reference construction; model truncates long text to its token limit.",
            "Index array_bytes excludes text, Python objects, ONNX model and temporary peak allocations.",
            "Qrels may be incomplete; unjudged documents are treated as nonrelevant for standard metrics.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_weight": chosen, "validation": report["validation"], "report": str(args.out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/benchmarks/scifact"))
    parser.add_argument("--cache", type=Path, default=Path("data/benchmarks/semantic-cache"))
    parser.add_argument("--queries", type=int, default=160)
    parser.add_argument("--dev-queries", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--commit", default="working-tree")
    parser.add_argument("--out", type=Path, default=Path("data/experiments/residual/scifact-pilot.json"))
    arguments = parser.parse_args()
    if arguments.top_k < 1 or arguments.candidates < arguments.top_k:
        parser.error("Require candidates >= top-k > 0")
    main(arguments)
