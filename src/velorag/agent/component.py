from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class AgentComponentType(str, Enum):
    BEGIN = "begin"
    RETRIEVAL = "retrieval"
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


class BaseComponent(ABC):
    def __init__(self, component_id: str, name: str, component_type: AgentComponentType):
        self.id = component_id
        self.name = name
        self.type = component_type
        self.config: dict[str, Any] = {}
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


class BeginComponent(BaseComponent):
    def __init__(self, component_id: str = "begin", name: str = "Begin"):
        super().__init__(component_id, name, AgentComponentType.BEGIN)

    async def execute(self, context: AgentContext) -> AgentContext:
        return context


class EndComponent(BaseComponent):
    def __init__(self, component_id: str = "end", name: str = "End"):
        super().__init__(component_id, name, AgentComponentType.END)

    async def execute(self, context: AgentContext) -> AgentContext:
        return context


class RetrievalComponent(BaseComponent):
    def __init__(self, component_id: str, name: str = "Retrieval", kb_id: str = "", top_k: int = 5):
        super().__init__(component_id, name, AgentComponentType.RETRIEVAL)
        self.config["kb_id"] = kb_id
        self.config["top_k"] = top_k

    async def execute(self, context: AgentContext) -> AgentContext:
        from velorag.pipeline import VeloRAG
        rag = VeloRAG()
        results = rag.search(
            query=context.query,
            top_k=self.config.get("top_k", 5),
            mode=self.config.get("mode", "hybrid"),
        )
        context.results.append({
            "component": self.id,
            "type": "retrieval",
            "data": [
                {"chunk_id": r.chunk.id, "text": r.chunk.text, "score": r.score}
                for r in results
            ],
        })
        return context


class GenerateComponent(BaseComponent):
    def __init__(self, component_id: str, name: str = "Generate", system_prompt: str = ""):
        super().__init__(component_id, name, AgentComponentType.GENERATE)
        self.config["system_prompt"] = system_prompt

    async def execute(self, context: AgentContext) -> AgentContext:
        from velorag.generation.llm import LLMGenerator
        from velorag.core.types import GenerationRequest, Chunk as CoreChunk

        generator = LLMGenerator()
        chunks = [
            CoreChunk(id="", text=item["text"], doc_id="", chunk_index=0)
            for r in context.results
            for item in r.get("data", [])
        ]
        request = GenerationRequest(
            query=context.query,
            context_chunks=chunks,
            system_prompt=self.config.get("system_prompt"),
        )
        response = generator.generate(request, [])
        context.results.append({
            "component": self.id,
            "type": "generation",
            "data": {"content": response.content, "token_count": response.token_count},
        })
        return context
