from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from velorag.api.router import api_router
from velorag.auth.dependencies import get_current_user, require_user
from velorag.core.database import get_db, get_engine, init_db
from velorag.core.security import resolve_safe_path
from velorag.core.settings import get_settings
from velorag.db.models import User
from velorag.knowledge.access import authorize_kb_access, build_scoped_filters
from velorag.server.state import get_default_pipeline

logger = logging.getLogger(__name__)
SearchModeLiteral = Literal["dense", "sparse", "hybrid", "graph_enhanced"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    get_default_pipeline(seed_builtin=True)
    yield
    # 优雅停机：释放数据库连接池资源
    get_engine().dispose()


app = FastAPI(
    title="VeloRAG API",
    description="High-velocity production RAG framework combining Hybrid Retrieval, Knowledge Graph, Grounded Citations & Agent Canvas",
    version="0.3.0",
    lifespan=lifespan,
)

_settings = get_settings()
_cors_origins = list(_settings.server.cors_origins)
# 遵循 CORS 安全规范：当 allow_origins 包含通配符 '*' 时禁止开启 allow_credentials
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """捕获未预期异常并仅向客户端返回脱敏错误信息，避免泄漏内部堆栈与服务器路径。"""
    logger.exception("未处理的服务器异常: %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


app.include_router(api_router)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, max_length=512)
    kb_id: str = Field(default="default", min_length=1, max_length=64)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchModeLiteral = "hybrid"
    use_graph: bool = True
    kb_id: Optional[str] = Field(default=None, max_length=64)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    top_k: int = Field(default=5, ge=1, le=50)
    mode: SearchModeLiteral = "hybrid"
    use_graph: bool = True
    kb_id: Optional[str] = Field(default=None, max_length=64)
    system_prompt: Optional[str] = Field(default=None, max_length=4000)
    min_relevance_score: float = Field(default=0.12, ge=0.0, le=1.0)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    rag = get_default_pipeline()
    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "version": "0.3.0",
        "indexed_chunks": rag.document_count,
        "kg_nodes": rag.knowledge_graph.node_count,
        "kg_edges": rag.knowledge_graph.edge_count,
    }


@app.post("/ingest")
def ingest(
    request: IngestRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """受控本地文档入库：强制认证、校验目标知识库写权限并限制路径位于安全根目录内。"""
    authorize_kb_access(db, request.kb_id, user, write=True, allow_unregistered=True)

    settings = get_settings()
    allowed_root = Path(settings.server.allowed_ingest_root or settings.index.persist_dir)
    allowed_root.mkdir(parents=True, exist_ok=True)
    try:
        target_path = resolve_safe_path(allowed_root, request.path)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is outside the allowed ingest directory",
        ) from None

    rag = get_default_pipeline()
    try:
        count = rag.ingest(str(target_path), metadata={"kb_id": request.kb_id})
        return {"ingested_chunks": count, "total_chunks": rag.document_count}
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Requested document path was not found",
        ) from None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None
    except Exception:
        logger.exception("本地路径入库失败: kb_id=%s", request.kb_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to ingest document",
        ) from None


@app.post("/search")
def search(
    request: SearchRequest,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = get_default_pipeline()
    scoped_filters = build_scoped_filters(db, user, kb_id=request.kb_id)
    results = rag.search(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        use_graph=request.use_graph,
        filters=scoped_filters or None,
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
def query(
    request: QueryRequest,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rag = get_default_pipeline()
    scoped_filters = build_scoped_filters(db, user, kb_id=request.kb_id)
    response = rag.query(
        query=request.query,
        top_k=request.top_k,
        mode=request.mode,
        use_graph=request.use_graph,
        system_prompt=request.system_prompt,
        min_relevance_score=request.min_relevance_score,
        filters=scoped_filters or None,
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
