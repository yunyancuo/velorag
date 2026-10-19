from __future__ import annotations

from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from velorag.auth.routes import router as auth_router
from velorag.knowledge.routes import router as kb_router
from velorag.chat.routes import router as chat_router
from velorag.agent.routes import router as agent_router
from velorag.eval.routes import router as eval_router
from velorag.server.state import get_default_pipeline

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(kb_router)
api_router.include_router(chat_router)
api_router.include_router(agent_router)
api_router.include_router(eval_router)


class RAGSearchPayload(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"
    use_graph: bool = True
    kb_id: Optional[str] = None


class RAGQueryPayload(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"
    use_graph: bool = True
    system_prompt: Optional[str] = None
    min_relevance_score: float = 0.12


class IngestTextPayload(BaseModel):
    title: str
    text: str
    source: str = "manual_input.md"
    kb_id: str = "default"


@api_router.post("/search", tags=["Search & QA"])
def api_search(req: RAGSearchPayload):
    rag = get_default_pipeline()
    filters = {"kb_id": req.kb_id} if req.kb_id else None
    results = rag.search(
        query=req.query,
        top_k=req.top_k,
        mode=req.mode,
        use_graph=req.use_graph,
        filters=filters,
    )
    return {
        "query": req.query,
        "mode": req.mode,
        "results": [
            {
                "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "text": r.chunk.text,
                "score": round(r.score, 4),
                "source": r.chunk.metadata.get("source", r.retrieval_source),
                "retrieval_source": r.retrieval_source,
                "metadata": r.chunk.metadata,
            }
            for r in results
        ],
    }


@api_router.post("/query", tags=["Search & QA"])
def api_query(req: RAGQueryPayload):
    rag = get_default_pipeline()
    resp = rag.query(
        query=req.query,
        top_k=req.top_k,
        mode=req.mode,
        use_graph=req.use_graph,
        system_prompt=req.system_prompt,
        min_relevance_score=req.min_relevance_score,
    )
    return {
        "answer": resp.content,
        "model": resp.model,
        "token_count": resp.token_count,
        "refused": resp.refused,
        "refusal_reason": resp.refusal_reason,
        "confidence": resp.confidence,
        "citations": [
            {
                "index": c.index,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "quote": c.quote,
                "score": c.score,
            }
            for c in resp.citations
        ],
        "sources": [
            {
                "chunk_id": s.chunk.id,
                "doc_id": s.chunk.doc_id,
                "text": s.chunk.text,
                "score": round(s.score, 4),
                "source": s.chunk.metadata.get("source", s.retrieval_source),
                "retrieval_source": s.retrieval_source,
            }
            for s in resp.sources
        ],
    }


@api_router.post("/ingest-text", tags=["Search & QA"])
def api_ingest_text(req: IngestTextPayload):
    rag = get_default_pipeline()
    chunks_added = rag.ingest_text(
        text=req.text,
        source=req.source or req.title,
        metadata={"title": req.title, "kb_id": req.kb_id},
    )
    return {
        "status": "indexed",
        "chunks_added": chunks_added,
        "total_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }


@api_router.get("/graph", tags=["Knowledge Graph"])
def api_get_graph(query: Optional[str] = None, limit: int = 50):
    rag = get_default_pipeline()
    return rag.knowledge_graph.get_subgraph(query=query, max_hops=2, limit=limit)


@api_router.get("/stats", tags=["System"])
def api_get_stats():
    rag = get_default_pipeline()
    return {
        "version": "0.3.0",
        "indexed_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }
