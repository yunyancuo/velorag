from __future__ import annotations

import json
import sys
import click
import uvicorn

from velorag.core.config import VeloRAGConfig
from velorag.pipeline import VeloRAG


@click.group()
@click.version_option(version="0.3.0")
def main():
    """VeloRAG: High-velocity hybrid RAG + Knowledge Graph + Agent Canvas framework."""
    pass


@main.command()
@click.argument("path")
@click.option("--config", "-c", default=None, help="Path to config YAML file")
def ingest(path: str, config: str | None):
    """Ingest documents from a file or directory."""
    cfg = VeloRAGConfig.from_yaml(config) if config else VeloRAGConfig()
    rag = VeloRAG(config=cfg)
    count = rag.ingest(path)
    click.echo(f"Ingested {count} chunks from {path} (total: {rag.document_count})")


@main.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results")
@click.option("--mode", "-m", default="hybrid", type=click.Choice(["dense", "sparse", "hybrid", "graph_enhanced", "local_residual", "residual_hybrid"]))
@click.option("--no-graph", is_flag=True, help="Disable knowledge graph enhancement")
def search(query: str, top_k: int, mode: str, no_graph: bool):
    """Search indexed documents."""
    rag = VeloRAG()
    results = rag.search(query=query, top_k=top_k, mode=mode, use_graph=not no_graph)
    for i, r in enumerate(results, 1):
        click.echo(f"\n--- [{i}] score={r.score:.4f} source={r.retrieval_source} ---")
        click.echo(r.chunk.text[:300])


@main.command()
@click.argument("question")
@click.option("--top-k", "-k", default=5, help="Number of context chunks")
@click.option("--mode", "-m", default="hybrid", type=click.Choice(["dense", "sparse", "hybrid", "graph_enhanced", "local_residual", "residual_hybrid"]))
def query(question: str, top_k: int, mode: str):
    """Ask a question with RAG-powered grounded generation."""
    rag = VeloRAG()
    response = rag.query(query=question, top_k=top_k, mode=mode)
    click.echo(f"\nAnswer (refused={response.refused}, confidence={response.confidence:.2f}):\n{response.content}")
    if response.citations:
        click.echo(f"\nCitations ({len(response.citations)}):")
        for c in response.citations:
            click.echo(f"  - [Source {c.index}] {c.source} (score={c.score:.3f}): {c.quote[:80]}")


@main.command(name="eval")
@click.option("--top-k", "-k", default=3, help="Top-K cutoff for retrieval metrics")
@click.option("--json-out", default=None, help="Optional file path to save full JSON evaluation report")
def evaluate_cmd(top_k: int, json_out: str | None):
    """Run the built-in Campus & AI Lab RAG + Citation + Refusal benchmark."""
    from velorag.eval.runner import EvaluationRunner

    runner = EvaluationRunner(top_k=top_k)
    suite = runner.run_suite()
    click.echo(f"\n=== {suite['benchmark_name']} ===")
    click.echo(f"Documents: {suite['document_count']} | Queries: {suite['sample_count']} | Top-K: {suite['top_k']}\n")
    click.echo(suite["markdown_table"])
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(suite, f, ensure_ascii=False, indent=2)
        click.echo(f"\nSaved detailed JSON report to {json_out}")


@main.command()
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
@click.option("--port", "-p", default=8000, help="Server port")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the VeloRAG REST API server."""
    uvicorn.run("velorag.server.app:app", host=host, port=port, reload=reload)


@main.command()
def info():
    """Display current configuration and index stats."""
    rag = VeloRAG()
    click.echo("VeloRAG v0.3.0")
    click.echo(f"Indexed chunks: {rag.document_count}")
    click.echo(f"KG nodes: {rag.knowledge_graph.node_count}")
    click.echo(f"KG edges: {rag.knowledge_graph.edge_count}")


if __name__ == "__main__":
    main()
