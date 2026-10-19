from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from velorag.agent.component import (
    AgentComponentType,
    AgentContext,
    BaseComponent,
    BeginComponent,
    CodeComponent,
    EndComponent,
    GenerateComponent,
    RerankComponent,
    RetrievalComponent,
    SwitchComponent,
    ToolComponent,
)


class AgentCanvas:
    """DAG-capable Agent workflow canvas supporting linear chains, conditional branching, and execution tracing."""

    def __init__(self, name: str, description: str = "", canvas_id: Optional[str] = None):
        self.id = canvas_id or uuid.uuid4().hex[:12]
        self.name = name
        self.description = description
        self._begin: BeginComponent = BeginComponent()
        self._current: Optional[BaseComponent] = self._begin
        self._nodes: dict[str, BaseComponent] = {self._begin.id: self._begin}
        self._edges: list[dict[str, str]] = []

    def add_component(self, component: BaseComponent, from_id: Optional[str] = None) -> BaseComponent:
        self._nodes[component.id] = component
        source_node = self._nodes.get(from_id) if from_id else self._current
        if source_node is not None and source_node.id != component.id:
            source_node.then(component)
            self.add_edge(source_node.id, component.id)
        self._current = component
        return component

    def add_edge(self, source_id: str, target_id: str, label: str = "default"):
        edge = {"source": source_id, "target": target_id, "label": label}
        if edge not in self._edges:
            self._edges.append(edge)

    async def run(self, query: str, **kwargs) -> AgentContext:
        context = AgentContext(query=query, params=kwargs)
        current_id: Optional[str] = self._begin.id
        visited_steps = 0
        max_steps = max(20, len(self._nodes) * 3)

        while current_id is not None and visited_steps < max_steps:
            node = self._nodes.get(current_id)
            if node is None:
                break

            visited_steps += 1
            t0 = time.perf_counter()
            context = await node.execute(context)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

            context.trace.append({
                "step": visited_steps,
                "node_id": node.id,
                "node_name": node.name,
                "node_type": node.type.value,
                "elapsed_ms": elapsed_ms,
            })

            if isinstance(node, SwitchComponent):
                passed = bool(context.variables.get(f"switch_{node.id}", False))
                next_target = (
                    node.config.get("true_target")
                    if passed
                    else node.config.get("false_target")
                )
                if next_target and next_target in self._nodes:
                    current_id = next_target
                    continue

            if node.next is not None:
                current_id = node.next.id
            else:
                outgoing = [e["target"] for e in self._edges if e["source"] == node.id]
                current_id = outgoing[0] if outgoing else None

        return context

    def get_components(self) -> list[BaseComponent]:
        return list(self._nodes.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": self._edges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentCanvas:
        canvas = cls(
            name=data.get("name", "Untitled Canvas"),
            description=data.get("description", ""),
            canvas_id=data.get("id"),
        )
        canvas._nodes.clear()
        canvas._edges = []

        for n in data.get("nodes", []):
            ntype = n.get("type", "begin")
            nid = n.get("id", uuid.uuid4().hex[:8])
            name = n.get("name", ntype.title())
            cfg = n.get("config", {})
            pos = n.get("position", {"x": 120, "y": 160})

            if ntype == "begin":
                comp: BaseComponent = BeginComponent(nid, name, position=pos)
                canvas._begin = comp
            elif ntype == "retrieval":
                comp = RetrievalComponent(
                    nid,
                    name,
                    kb_id=cfg.get("kb_id", ""),
                    top_k=cfg.get("top_k", 5),
                    mode=cfg.get("mode", "hybrid"),
                    use_graph=cfg.get("use_graph", True),
                    position=pos,
                )
            elif ntype == "rerank":
                comp = RerankComponent(
                    nid,
                    name,
                    min_score=cfg.get("min_score", 0.1),
                    top_k=cfg.get("top_k", 5),
                    position=pos,
                )
            elif ntype == "switch":
                comp = SwitchComponent(
                    nid,
                    name,
                    variable=cfg.get("variable", "max_score"),
                    operator=cfg.get("operator", ">="),
                    threshold=cfg.get("threshold", 0.12),
                    true_target=cfg.get("true_target"),
                    false_target=cfg.get("false_target"),
                    position=pos,
                )
            elif ntype == "code":
                comp = CodeComponent(nid, name, code=cfg.get("code", ""), position=pos)
            elif ntype == "tool":
                comp = ToolComponent(nid, name, tool_name=cfg.get("tool_name", "verifier"), position=pos)
            elif ntype == "generate":
                comp = GenerateComponent(
                    nid,
                    name,
                    system_prompt=cfg.get("system_prompt", ""),
                    min_relevance_score=cfg.get("min_relevance_score", 0.12),
                    position=pos,
                )
            else:
                comp = EndComponent(nid, name, position=pos)

            comp.config.update(cfg)
            canvas._nodes[nid] = comp

        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            lbl = edge.get("label", "default")
            if src in canvas._nodes and tgt in canvas._nodes:
                canvas.add_edge(src, tgt, lbl)
                if canvas._nodes[src].next is None and lbl == "default":
                    canvas._nodes[src].then(canvas._nodes[tgt])

        return canvas

    @classmethod
    def create_default_rag_canvas(cls, kb_id: str = "default") -> AgentCanvas:
        """Create a pre-configured production RAG + Graph + Guardrail Switch + Citation workflow."""
        canvas = cls(
            name="Campus & Research Grounded RAG Workflow",
            description="Hybrid + Knowledge Graph retrieval -> Lexical Reranker -> Evidence Guardrail Switch -> Grounded Citation Generator",
            canvas_id="default-grounded-rag",
        )
        canvas._begin.position = {"x": 60, "y": 180}
        ret = RetrievalComponent("retrieve_1", "Hybrid + KG Retrieval", kb_id=kb_id, top_k=6, mode="graph_enhanced", position={"x": 240, "y": 180})
        rerank = RerankComponent("rerank_1", "Lexical Overlap Reranker", min_score=0.08, top_k=4, position={"x": 450, "y": 180})
        switch = SwitchComponent(
            "switch_1",
            "Evidence Sufficiency Check",
            variable="max_score",
            operator=">=",
            threshold=0.12,
            true_target="generate_1",
            false_target="generate_1",
            position={"x": 660, "y": 180},
        )
        gen = GenerateComponent("generate_1", "Grounded Citation Generator", min_relevance_score=0.12, position={"x": 880, "y": 180})
        end = EndComponent("end_1", "End", position={"x": 1080, "y": 180})

        canvas.add_component(ret)
        canvas.add_component(rerank)
        canvas.add_component(switch)
        canvas.add_component(gen)
        canvas.add_component(end)
        return canvas
