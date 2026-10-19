from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from velorag.agent.canvas import AgentCanvas

router = APIRouter(prefix="/agents", tags=["Agent Canvas"])

# In-memory registry seeded with default RAG workflow
_CANVAS_STORE: dict[str, AgentCanvas] = {
    "default-grounded-rag": AgentCanvas.create_default_rag_canvas("default"),
}


class CanvasSaveRequest(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, str]] = Field(default_factory=list)


class CanvasRunRequest(BaseModel):
    query: str
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
    if req.nodes:
        canvas = AgentCanvas.from_dict(req.model_dump())
    else:
        canvas = AgentCanvas.create_default_rag_canvas("default")
        canvas.name = req.name
        canvas.description = req.description
        if req.id:
            canvas.id = req.id
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
async def run_canvas(canvas_id: str, req: CanvasRunRequest):
    canvas = _CANVAS_STORE.get(canvas_id)
    if not canvas:
        raise HTTPException(status_code=404, detail="Agent canvas not found")

    from velorag.server.state import get_default_pipeline
    pipeline = get_default_pipeline()
    context = await canvas.run(req.query, pipeline=pipeline, **req.params)
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
