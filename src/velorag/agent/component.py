from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class AgentComponentType(str, Enum):
    BEGIN = "begin"
    RETRIEVAL = "retrieval"
    RERANK = "rerank"
    GENERATE = "generate"
    SWITCH = "switch"
    CODE = "code"
    TOOL = "tool"
    END = "end"


@dataclass
class AgentContext:
    query: str
    history: list[dict] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    refused: bool = False


class BaseComponent(ABC):
    def __init__(
        self,
        component_id: str,
        name: str,
        component_type: AgentComponentType,
        position: dict[str, float] | None = None,
    ):
        self.id = component_id
        self.name = name
        self.type = component_type
        self.config: dict[str, Any] = {}
        self.position: dict[str, float] = position or {"x": 120.0, "y": 140.0}
        self._next: Optional[BaseComponent] = None

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentContext:
        ...

    def then(self, component: BaseComponent) -> BaseComponent:
        self._next = component
        return component

    @property
    def next(self) -> Optional[BaseComponent]:
        return self._next

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "config": self.config,
            "position": self.position,
        }


class BeginComponent(BaseComponent):
    def __init__(self, component_id: str = "begin", name: str = "Begin", position: dict[str, float] | None = None):
        super().__init__(component_id, name, AgentComponentType.BEGIN, position or {"x": 80, "y": 180})

    async def execute(self, context: AgentContext) -> AgentContext:
        context.variables["query"] = context.query
        return context


class EndComponent(BaseComponent):
    def __init__(self, component_id: str = "end", name: str = "End", position: dict[str, float] | None = None):
        super().__init__(component_id, name, AgentComponentType.END, position or {"x": 880, "y": 180})

    async def execute(self, context: AgentContext) -> AgentContext:
        if not context.final_answer:
            for item in reversed(context.results):
                if item.get("type") == "generation":
                    context.final_answer = item.get("data", {}).get("content", "")
                    break
        return context


class RetrievalComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        name: str = "Hybrid Retrieval",
        kb_id: str = "",
        top_k: int = 5,
        mode: str = "hybrid",
        use_graph: bool = True,
        pipeline: Any = None,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.RETRIEVAL, position or {"x": 280, "y": 180})
        self.config["kb_id"] = kb_id
        self.config["top_k"] = top_k
        self.config["mode"] = mode
        self.config["use_graph"] = use_graph
        self._pipeline = pipeline

    async def execute(self, context: AgentContext) -> AgentContext:
        rag = self._pipeline or context.params.get("pipeline")
        if rag is None:
            from velorag.pipeline import VeloRAG
            rag = VeloRAG()

        results = rag.search(
            query=context.query,
            top_k=int(self.config.get("top_k", 5)),
            mode=str(self.config.get("mode", "hybrid")),
            use_graph=bool(self.config.get("use_graph", True)),
        )
        serialized = [
            {
                "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "text": r.chunk.text,
                "score": r.score,
                "source": r.chunk.metadata.get("source", r.retrieval_source),
                "retrieval_source": r.retrieval_source,
                "metadata": r.chunk.metadata,
            }
            for r in results
        ]
        context.variables["retrieved_chunks"] = serialized
        context.variables["hit_count"] = len(serialized)
        context.variables["max_score"] = max((r["score"] for r in serialized), default=0.0)
        context.results.append({
            "component": self.id,
            "type": "retrieval",
            "data": serialized,
        })
        return context


class RerankComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        name: str = "Lexical Reranker",
        min_score: float = 0.1,
        top_k: int = 5,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.RERANK, position or {"x": 480, "y": 180})
        self.config["min_score"] = min_score
        self.config["top_k"] = top_k

    async def execute(self, context: AgentContext) -> AgentContext:
        from velorag.core.types import Chunk, SearchResult
        from velorag.retrieval.reranker import LexicalOverlapReranker

        raw_items = context.variables.get("retrieved_chunks", [])
        search_results = [
            SearchResult(
                chunk=Chunk(
                    id=item.get("chunk_id", ""),
                    text=item.get("text", ""),
                    doc_id=item.get("doc_id", ""),
                    chunk_index=i,
                    metadata=item.get("metadata", {}),
                ),
                score=float(item.get("score", 0.0)),
                retrieval_source=item.get("retrieval_source", "hybrid"),
            )
            for i, item in enumerate(raw_items)
        ]
        reranker = LexicalOverlapReranker()
        reranked = reranker.rerank(context.query, search_results)
        min_score = float(self.config.get("min_score", 0.1))
        top_k = int(self.config.get("top_k", 5))
        filtered = [r for r in reranked if r.score >= min_score][:top_k]

        serialized = [
            {
                "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "text": r.chunk.text,
                "score": r.score,
                "source": r.chunk.metadata.get("source", r.retrieval_source),
                "retrieval_source": r.retrieval_source,
                "metadata": r.chunk.metadata,
            }
            for r in filtered
        ]
        context.variables["retrieved_chunks"] = serialized
        context.variables["hit_count"] = len(serialized)
        context.variables["max_score"] = max((r["score"] for r in serialized), default=0.0)
        context.results.append({
            "component": self.id,
            "type": "rerank",
            "data": serialized,
        })
        return context


class SwitchComponent(BaseComponent):
    """Conditional routing node based on context variables (e.g. hit_count > 0 or max_score >= threshold)."""

    def __init__(
        self,
        component_id: str,
        name: str = "Evidence Guardrail Switch",
        variable: str = "max_score",
        operator: str = ">=",
        threshold: float = 0.12,
        true_target: Optional[str] = None,
        false_target: Optional[str] = None,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.SWITCH, position or {"x": 520, "y": 180})
        self.config["variable"] = variable
        self.config["operator"] = operator
        self.config["threshold"] = threshold
        self.config["true_target"] = true_target
        self.config["false_target"] = false_target

    def evaluate_condition(self, context: AgentContext) -> bool:
        var_name = self.config.get("variable", "max_score")
        val = context.variables.get(var_name, 0.0)
        op = self.config.get("operator", ">=")
        thresh = self.config.get("threshold", 0.12)
        if op == ">=":
            return float(val) >= float(thresh)
        if op == ">":
            return float(val) > float(thresh)
        if op == "<=":
            return float(val) <= float(thresh)
        if op == "<":
            return float(val) < float(thresh)
        if op == "==":
            return val == thresh
        return bool(val)

    async def execute(self, context: AgentContext) -> AgentContext:
        passed = self.evaluate_condition(context)
        context.variables[f"switch_{self.id}"] = passed
        context.results.append({
            "component": self.id,
            "type": "switch",
            "data": {
                "condition_passed": passed,
                "selected_target": self.config.get("true_target") if passed else self.config.get("false_target"),
            },
        })
        return context


class CodeComponent(BaseComponent):
    """Lightweight transformation component for custom context post-processing."""

    def __init__(
        self,
        component_id: str,
        name: str = "Code Transform",
        code: str = "output = {'summary_count': len(variables.get('retrieved_chunks', []))}",
        fn: Optional[Callable[[AgentContext], dict[str, Any]]] = None,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.CODE, position or {"x": 560, "y": 300})
        self.config["code"] = code
        self._fn = fn

    async def execute(self, context: AgentContext) -> AgentContext:
        if self._fn is not None:
            out = self._fn(context)
        else:
            local_env: dict[str, Any] = {
                "query": context.query,
                "variables": context.variables,
                "output": {},
            }
            safe_builtins = {"len": len, "min": min, "max": max, "sum": sum, "sorted": sorted, "str": str, "int": int, "float": float, "bool": bool, "dict": dict, "list": list}
            exec(self.config.get("code", ""), {"__builtins__": safe_builtins}, local_env)
            out = local_env.get("output", {})

        if isinstance(out, dict):
            context.variables.update(out)
        context.results.append({
            "component": self.id,
            "type": "code",
            "data": out,
        })
        return context


class ToolComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        name: str = "Tool Invocation",
        tool_name: str = "citation_verifier",
        tool_fn: Optional[Callable[[AgentContext], Any]] = None,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.TOOL, position or {"x": 620, "y": 300})
        self.config["tool_name"] = tool_name
        self._tool_fn = tool_fn

    async def execute(self, context: AgentContext) -> AgentContext:
        if self._tool_fn:
            res = self._tool_fn(context)
        else:
            chunks = context.variables.get("retrieved_chunks", [])
            res = {
                "tool": self.config.get("tool_name", "citation_verifier"),
                "verified_sources": len(chunks),
                "status": "verified" if chunks else "empty",
            }
        context.results.append({
            "component": self.id,
            "type": "tool",
            "data": res,
        })
        return context


class GenerateComponent(BaseComponent):
    def __init__(
        self,
        component_id: str,
        name: str = "Grounded Generate",
        system_prompt: str = "",
        min_relevance_score: float = 0.12,
        position: dict[str, float] | None = None,
    ):
        super().__init__(component_id, name, AgentComponentType.GENERATE, position or {"x": 680, "y": 180})
        self.config["system_prompt"] = system_prompt
        self.config["min_relevance_score"] = min_relevance_score

    async def execute(self, context: AgentContext) -> AgentContext:
        from velorag.generation.llm import LLMGenerator
        from velorag.core.types import GenerationRequest, Chunk as CoreChunk, SearchResult

        generator = LLMGenerator()
        raw_chunks = context.variables.get("retrieved_chunks")
        if raw_chunks is None:
            raw_chunks = [
                item
                for r in context.results
                if r.get("type") in ("retrieval", "rerank")
                for item in r.get("data", [])
            ]

        chunks: list[CoreChunk] = []
        search_results: list[SearchResult] = []
        for idx, item in enumerate(raw_chunks):
            c = CoreChunk(
                id=item.get("chunk_id") or f"c_{idx + 1}",
                text=item.get("text", ""),
                doc_id=item.get("doc_id", ""),
                chunk_index=idx,
                metadata=item.get("metadata") or {"source": item.get("source", f"source_{idx + 1}")},
            )
            chunks.append(c)
            search_results.append(
                SearchResult(
                    chunk=c,
                    score=float(item.get("score", 0.5)),
                    retrieval_source=item.get("retrieval_source", "hybrid"),
                )
            )

        request = GenerationRequest(
            query=context.query,
            context_chunks=chunks,
            system_prompt=self.config.get("system_prompt") or None,
            min_relevance_score=float(self.config.get("min_relevance_score", 0.12)),
        )
        response = generator.generate(request, search_results)
        citation_dicts = [
            {
                "index": c.index,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "quote": c.quote,
                "score": c.score,
            }
            for c in response.citations
        ]
        context.final_answer = response.content
        context.citations = citation_dicts
        context.refused = response.refused
        context.results.append({
            "component": self.id,
            "type": "generation",
            "data": {
                "content": response.content,
                "token_count": response.token_count,
                "citations": citation_dicts,
                "refused": response.refused,
                "refusal_reason": response.refusal_reason,
                "confidence": response.confidence,
            },
        })
        return context
