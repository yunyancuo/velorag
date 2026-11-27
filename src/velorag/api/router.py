from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from velorag.agent.routes import router as agent_router
from velorag.auth.dependencies import get_current_user
from velorag.auth.routes import router as auth_router
from velorag.chat.routes import router as chat_router
from velorag.core.database import get_db
from velorag.db.models import User
from velorag.eval.routes import router as eval_router
from velorag.knowledge.access import authorize_kb_access, build_scoped_filters
from velorag.knowledge.routes import router as kb_router
from velorag.server.state import get_default_pipeline

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(auth_router)
api_router.include_router(kb_router)
api_router.include_router(chat_router)
api_router.include_router(agent_router)
api_router.include_router(eval_router)

SearchModeLiteral = Literal["dense", "sparse", "hybrid", "graph_enhanced"]


class RAGSearchPayload(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchModeLiteral = "hybrid"
    use_graph: bool = True
    kb_id: Optional[str] = Field(default=None, max_length=64)
    filters: Optional[dict[str, Any]] = None


class RAGQueryPayload(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchModeLiteral = "hybrid"
    use_graph: bool = True
    kb_id: Optional[str] = Field(default=None, max_length=64)
    filters: Optional[dict[str, Any]] = None
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    min_relevance_score: float = Field(default=0.12, ge=0.0, le=1.0)


class IngestTextPayload(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=2_000_000)
    source: str = Field(default="manual_input.md", max_length=256)
    kb_id: str = Field(default="default", min_length=1, max_length=64)


@api_router.post("/search", tags=["Search & QA"])
def api_search(
    req: RAGSearchPayload,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = get_default_pipeline()
    scoped_filters = build_scoped_filters(
        db, user, kb_id=req.kb_id, extra_filters=req.filters
    )
    results = rag.search(
        query=req.query,
        top_k=req.top_k,
        mode=req.mode,
        use_graph=req.use_graph,
        filters=scoped_filters or None,
    )
    return {
        "query": req.query,
        "mode": req.mode,
        "kb_id": req.kb_id,
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
def api_query(
    req: RAGQueryPayload,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = get_default_pipeline()
    scoped_filters = build_scoped_filters(
        db, user, kb_id=req.kb_id, extra_filters=req.filters
    )
    resp = rag.query(
        query=req.query,
        top_k=req.top_k,
        mode=req.mode,
        use_graph=req.use_graph,
        system_prompt=req.system_prompt,
        min_relevance_score=req.min_relevance_score,
        filters=scoped_filters or None,
    )
    return {
        "answer": resp.content,
        "model": resp.model,
        "token_count": resp.token_count,
        "refused": resp.refused,
        "refusal_reason": resp.refusal_reason,
        "confidence": resp.confidence,
        "kb_id": req.kb_id,
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
def api_ingest_text(
    req: IngestTextPayload,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authorize_kb_access(db, req.kb_id, user, write=True, allow_unregistered=True)
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
def api_get_graph(
    query: Optional[str] = None,
    kb_id: Optional[str] = None,
    max_hops: int = Query(default=2, ge=0, le=4),
    limit: int = Query(default=50, ge=1, le=200),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = get_default_pipeline()
    scoped_filters = build_scoped_filters(db, user, kb_id=kb_id)
    allowed_chunk_ids = (
        rag.get_chunk_ids_for_filters(scoped_filters) if scoped_filters else None
    )
    return rag.knowledge_graph.get_subgraph(
        query=query,
        max_hops=max_hops,
        limit=limit,
        allowed_chunk_ids=allowed_chunk_ids,
    )


@api_router.get("/stats", tags=["System"])
def api_get_stats():
    rag = get_default_pipeline()
    return {
        "version": "0.3.0",
        "indexed_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }
