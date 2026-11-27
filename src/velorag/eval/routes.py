from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from velorag.eval.dataset import get_builtin_campus_benchmark
from velorag.eval.runner import EvaluationRunner

router = APIRouter(prefix="/eval", tags=["RAG Evaluation"])

EvalModeLiteral = Literal["dense", "sparse", "hybrid", "graph_enhanced"]


class EvalRunRequest(BaseModel):
    modes: list[EvalModeLiteral] = Field(
        default_factory=lambda: ["dense", "sparse", "hybrid", "graph_enhanced"],
        min_length=1,
        max_length=4,
    )
    top_k: int = Field(default=3, ge=1, le=20)


@router.get("/dataset")
def get_eval_dataset():
    docs, samples = get_builtin_campus_benchmark()
    return {
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "text": d.text, "metadata": d.metadata}
            for d in docs
        ],
        "samples": [
            {
                "id": s.id,
                "query": s.query,
                "relevant_doc_ids": s.relevant_doc_ids,
                "expected_keywords": s.expected_keywords,
                "should_refuse": s.should_refuse,
                "category": s.category,
            }
            for s in samples
        ],
    }


@router.post("/run")
def run_evaluation(req: Optional[EvalRunRequest] = None):
    request = req or EvalRunRequest()
    runner = EvaluationRunner(top_k=request.top_k)
    return runner.run_suite(modes=request.modes)
