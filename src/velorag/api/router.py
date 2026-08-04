from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from velorag.auth.dependencies import require_user
from velorag.auth.routes import router as auth_router
from velorag.knowledge.routes import router as kb_router
from velorag.server.app import router as search_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(kb_router)
api_router.include_router(search_router)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


@api_router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse()
