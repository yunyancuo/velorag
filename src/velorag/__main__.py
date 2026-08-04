#!/usr/bin/env python
"""VeloRAG CLI - High-performance RAG framework."""

import argparse
import sys
from pathlib import Path


def cmd_serve(args):
    from velorag.server.app import run_server
    run_server(host=args.host, port=args.port)


def cmd_ingest(args):
    from velorag.pipeline import VeloRAG
    rag = VeloRAG()

    path = Path(args.path)
    if path.is_file():
        chunks = rag.ingest_file(str(path))
        print(f"Indexed {len(chunks)} chunks from {path.name}")
    elif path.is_dir():
        chunks = rag.ingest_directory(str(path), recursive=not args.no_recursive)
        print(f"Indexed {len(chunks)} chunks from directory {path}")
    else:
        print(f"Path not found: {path}", file=sys.stderr)
        sys.exit(1)


def cmd_search(args):
    from velorag.pipeline import VeloRAG
    rag = VeloRAG()

    results = rag.search(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        use_graph=not args.no_graph,
    )

    for i, r in enumerate(results):
        print(f"\n--- Result {i + 1} [{r.score:.3f}] ({r.retrieval_source}) ---")
        print(r.chunk.text[:300])
        if len(r.chunk.text) > 300:
            print("...")


def cmd_ask(args):
    from velorag.pipeline import VeloRAG
    rag = VeloRAG()

    result = rag.query(
        query=args.query,
        top_k=args.top_k,
        mode=args.mode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print(f"\nAnswer:\n{result.content}\n")
    print(f"--- Tokens: {result.token_count} | Sources: {len(result.sources)} ---")


def cmd_status(args):
    from velorag.pipeline import VeloRAG
    rag = VeloRAG()
    print(f"Chunks indexed: {rag.vector_store.count() if rag.vector_store else 0}")
    if rag.knowledge_graph:
        print(f"Entities: {rag.knowledge_graph.get_entity_count()}")
        print(f"Relations: {rag.knowledge_graph.get_relation_count()}")


def main():
    parser = argparse.ArgumentParser(description="VeloRAG - High-performance RAG Framework")
    sub = parser.add_subparsers(dest="command", help="Commands")

    p_serve = sub.add_parser("serve", help="Start the API server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_ingest = sub.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.add_argument("--no-recursive", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Search the knowledge base")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--mode", default="hybrid", choices=["dense", "sparse", "hybrid", "graph_enhanced"])
    p_search.add_argument("--no-graph", action="store_true")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Ask a question with RAG")
    p_ask.add_argument("query", help="Question to ask")
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--mode", default="hybrid")
    p_ask.add_argument("--max-tokens", type=int, default=2048)
    p_ask.add_argument("--temperature", type=float, default=0.7)
    p_ask.set_defaults(func=cmd_ask)

    p_status = sub.add_parser("status", help="Show index status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
