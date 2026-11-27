from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from velorag.agent.canvas import AgentCanvas
from velorag.agent.component import RetrievalComponent
from velorag.auth.dependencies import get_current_user
from velorag.core.database import get_db
from velorag.db.models import User
from velorag.knowledge.access import authorize_kb_access, build_scoped_filters

router = APIRouter(prefix="/agents", tags=["Agent Canvas"])

# 内存工作流注册表（预置默认 Grounded RAG 模板）
_CANVAS_STORE: dict[str, AgentCanvas] = {
    "default-grounded-rag": AgentCanvas.create_default_rag_canvas("default"),
}


class CanvasSaveRequest(BaseModel):
    id: Optional[str] = Field(default=None, max_length=64)
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=1000)
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)


class CanvasRunRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    kb_id: Optional[str] = Field(default=None, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


@router.get("")
def list_canvases():
    return [c.to_dict() for c in _CANVAS_STORE.values()]


@router.get("/{canvas_id}")
def get_canvas(canvas_id: str):
    canvas = _CANVAS_STORE.get(canvas_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Agent canvas not found")
    return canvas.to_dict()


@router.post("")
def create_or_save_canvas(req: CanvasSaveRequest):
    try:
        if req.nodes:
            canvas = AgentCanvas.from_dict(req.model_dump())
            canvas.validate(raise_on_error=True)
        else:
            canvas = AgentCanvas.create_default_rag_canvas("default")
            canvas.name = req.name
            canvas.description = req.description
            if req.id:
                canvas.id = req.id
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    _CANVAS_STORE[canvas.id] = canvas
    return canvas.to_dict()


@router.delete("/{canvas_id}")
def delete_canvas(canvas_id: str):
    if canvas_id not in _CANVAS_STORE:
        raise HTTPException(status_code=404, detail="Agent canvas not found")
    if canvas_id == "default-grounded-rag":
        raise HTTPException(status_code=400, detail="Cannot delete default workflow template")
    del _CANVAS_STORE[canvas_id]
    return {"status": "deleted", "id": canvas_id}


@router.post("/{canvas_id}/run")
async def run_canvas(
    canvas_id: str,
    req: CanvasRunRequest,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    canvas = _CANVAS_STORE.get(canvas_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Agent canvas not found")

    run_params = dict(req.params)
    # 阻止客户端覆盖内部 pipeline 实例
    run_params.pop("pipeline", None)

    explicit_kb_id = req.kb_id or run_params.get("kb_id")
    if explicit_kb_id:
        explicit_kb_id = str(explicit_kb_id)
        scoped_filters = build_scoped_filters(
            db,
            user,
            kb_id=explicit_kb_id,
            extra_filters=run_params.get("filters") if isinstance(run_params.get("filters"), dict) else None,
        )
        run_params["kb_id"] = explicit_kb_id
        run_params["filters"] = scoped_filters
    else:
        # 检查画布中所有 RetrievalComponent 配置的 kb_id 权限
        configured_kbs = {
            str(comp.config.get("kb_id"))
            for comp in canvas.get_components()
            if isinstance(comp, RetrievalComponent) and comp.config.get("kb_id")
        }
        for kb_id in configured_kbs:
            authorize_kb_access(db, kb_id, user, write=False, allow_unregistered=True)

        if not configured_kbs:
            scoped_filters = build_scoped_filters(
                db,
                user,
                kb_id=None,
                extra_filters=run_params.get("filters") if isinstance(run_params.get("filters"), dict) else None,
            )
            if scoped_filters:
                run_params["filters"] = scoped_filters

    from velorag.server.state import get_default_pipeline

    pipeline = get_default_pipeline()
    try:
        context = await canvas.run(req.query, pipeline=pipeline, **run_params)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    return {
        "canvas_id": canvas.id,
        "query": context.query,
        "final_answer": context.final_answer,
        "citations": context.citations,
        "refused": context.refused,
        "variables": {
            k: v for k, v in context.variables.items() if k != "retrieved_chunks"
        },
        "trace": context.trace,
        "results": context.results,
    }
