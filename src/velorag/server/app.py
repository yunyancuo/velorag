from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from velorag.api.router import api_router
from velorag.core.database import init_db
from velorag.server.state import get_default_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_default_pipeline(seed_builtin=True)
    yield


app = FastAPI(
    title="VeloRAG API",
    description="High-velocity production RAG framework combining Hybrid Retrieval, Knowledge Graph, Grounded Citations & Agent Canvas",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


class IngestRequest(BaseModel):
    path: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"
    use_graph: bool = True


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "hybrid"
    use_graph: bool = True
    system_prompt: Optional[str] = None
    min_relevance_score: float = 0.12


@app.get("/health")
def health():
    rag = get_default_pipeline()
    return {
        "status": "ok",
        "version": "0.3.0",
        "indexed_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }


@app.post("/ingest")
def ingest(request: IngestRequest):
    rag = get_default_pipeline()
    try:
        count = rag.ingest(request.path)
        return {"ingested_chunks": count, "total_chunks": rag.document_count}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(request: SearchRequest):
    rag = get_default_pipeline()
    results = rag.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        use_graph=request.use_graph,
    )
    return {
        "results": [
            {
                "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "text": r.chunk.text,
                "score": r.score,
                "source": r.retrieval_source,
                "metadata": r.chunk.metadata,
            }
            for r in results
        ]
    }


@app.post("/query")
def query(request: QueryRequest):
    rag = get_default_pipeline()
    response = rag.query(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        use_graph=request.use_graph,
        system_prompt=request.system_prompt,
        min_relevance_score=request.min_relevance_score,
    )
    return {
        "answer": response.content,
        "model": response.model,
        "token_count": response.token_count,
        "refused": response.refused,
        "refusal_reason": response.refusal_reason,
        "confidence": response.confidence,
        "citations": [
            {
                "index": c.index,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "quote": c.quote,
                "score": c.score,
            }
            for c in response.citations
        ],
        "sources": [
            {
                "chunk_id": s.chunk.id,
                "doc_id": s.chunk.doc_id,
                "text": s.chunk.text[:240],
                "score": s.score,
                "source": s.retrieval_source,
            }
            for s in response.sources
        ],
    }


@app.get("/stats")
def stats():
    rag = get_default_pipeline()
    return {
        "indexed_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }
