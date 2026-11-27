from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Any, Optional

from velorag.agent.component import (
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
    validate_safe_code,
)


class AgentCanvas:
    """支持 DAG 环路检测、孤立节点诊断、条件分支与全链路追踪的 Agent 工作流画布。"""

    def __init__(self, name: str, description: str = "", canvas_id: Optional[str] = None):
        self.id = canvas_id or uuid.uuid4().hex[:12]
        self.name = name
        self.description = description
        self._begin: Optional[BeginComponent] = BeginComponent()
        self._current: Optional[BaseComponent] = self._begin
        self._nodes: dict[str, BaseComponent] = {self._begin.id: self._begin}
        self._edges: list[dict[str, str]] = []

    def add_component(self, component: BaseComponent, from_id: Optional[str] = None) -> BaseComponent:
        self._nodes[component.id] = component
        if isinstance(component, BeginComponent):
            self._begin = component
        source_node = self._nodes.get(from_id) if from_id else self._current
        if source_node is not None and source_node.id != component.id:
            source_node.then(component)
            self.add_edge(source_node.id, component.id)
        self._current = component
        return component

    def add_edge(self, source_id: str, target_id: str, label: str = "default"):
        if source_id == target_id:
            raise ValueError(f"AgentCanvas 不允许自环边: {source_id} -> {target_id}")
        edge = {"source": source_id, "target": target_id, "label": label}
        if edge not in self._edges:
            self._edges.append(edge)

    def _build_adjacency(self) -> dict[str, list[str]]:
        """聚合显式边、链式 next 指针与 Switch 条件分支目标，构造完整有向图邻接表。"""
        adj: dict[str, list[str]] = {nid: [] for nid in self._nodes}

        def _append(src: str, tgt: Optional[str]) -> None:
            if src in adj and tgt and tgt not in adj[src]:
                adj[src].append(tgt)

        for edge in self._edges:
            _append(edge["source"], edge["target"])

        for nid, node in self._nodes.items():
            if node.next is not None:
                _append(nid, node.next.id)
            if isinstance(node, SwitchComponent):
                _append(nid, node.config.get("true_target"))
                _append(nid, node.config.get("false_target"))
        return adj

    def validate(self, raise_on_error: bool = False) -> dict[str, Any]:
        """执行 DAG 完整性校验：检测缺失起始节点、悬空边、环路、危险代码及孤立节点。"""
        errors: list[str] = []
        warnings: list[str] = []

        if self._begin is None or self._begin.id not in self._nodes:
            errors.append("画布缺少有效的 Begin 起始节点")

        # 1. 校验悬空边与 CodeComponent 安全语法
        for edge in self._edges:
            if edge["source"] not in self._nodes or edge["target"] not in self._nodes:
                errors.append(
                    f"存在指向未知节点的悬空边: {edge['source']} -> {edge['target']}"
                )

        for nid, node in self._nodes.items():
            if isinstance(node, SwitchComponent):
                for branch_key in ("true_target", "false_target"):
                    tgt = node.config.get(branch_key)
                    if tgt and tgt not in self._nodes:
                        errors.append(
                            f"Switch 节点 {nid} 的 {branch_key} 指向不存在的节点: {tgt}"
                        )
            elif isinstance(node, CodeComponent) and node._fn is None:
                code_str = str(node.config.get("code", ""))
                if code_str.strip():
                    try:
                        validate_safe_code(code_str)
                    except ValueError as exc:
                        errors.append(f"Code 节点 {nid} 安全校验失败: {exc}")

        # 2. 三色 DFS 检测 DAG 环路
        adj = self._build_adjacency()
        color: dict[str, int] = {nid: 0 for nid in self._nodes}  # 0=未访问, 1=访问中, 2=已完成
        path: list[str] = []

        def _dfs(u: str) -> None:
            color[u] = 1
            path.append(u)
            for v in adj.get(u, ()):
                if v not in color:
                    continue
                if color[v] == 1:
                    cycle_start = path.index(v)
                    cycle_chain = " -> ".join(path[cycle_start:] + [v])
                    errors.append(f"检测到 DAG 环路: {cycle_chain}")
                elif color[v] == 0:
                    _dfs(v)
            path.pop()
            color[u] = 2

        for nid in self._nodes:
            if color[nid] == 0:
                _dfs(nid)

        # 3. 孤立/不可达节点诊断
        unreachable_nodes: list[str] = []
        if self._begin is not None and self._begin.id in self._nodes:
            reachable: set[str] = {self._begin.id}
            queue: deque[str] = deque([self._begin.id])
            while queue:
                curr = queue.popleft()
                for nxt in adj.get(curr, ()):
                    if nxt in self._nodes and nxt not in reachable:
                        reachable.add(nxt)
                        queue.append(nxt)
            unreachable_nodes = [nid for nid in self._nodes if nid not in reachable]
            if unreachable_nodes:
                warnings.append(
                    f"存在不可达/孤立节点: {', '.join(unreachable_nodes)}"
                )

        report = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "unreachable_nodes": unreachable_nodes,
        }
        if raise_on_error and errors:
            raise ValueError("; ".join(errors))
        return report

    async def run(self, query: str, **kwargs) -> AgentContext:
        # 执行前强制做 DAG 环路与结构校验，彻底防止死循环
        self.validate(raise_on_error=True)

        context = AgentContext(query=query, params=kwargs)
        assert self._begin is not None
        current_id: Optional[str] = self._begin.id
        visited_nodes: set[str] = set()
        visited_steps = 0

        while current_id is not None:
            if current_id in visited_nodes:
                raise ValueError(f"运行期检测到节点重复访问形成环路: {current_id}")
            node = self._nodes.get(current_id)
            if node is None:
                break

            visited_nodes.add(current_id)
            visited_steps += 1
            t0 = time.perf_counter()
            try:
                context = await node.execute(context)
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
                context.trace.append(
                    {
                        "step": visited_steps,
                        "node_id": node.id,
                        "node_name": node.name,
                        "node_type": node.type.value,
                        "elapsed_ms": elapsed_ms,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                raise RuntimeError(
                    f"AgentCanvas 节点 '{node.id}' ({node.type.value}) 执行失败: {exc}"
                ) from exc

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            context.trace.append(
                {
                    "step": visited_steps,
                    "node_id": node.id,
                    "node_name": node.name,
                    "node_type": node.type.value,
                    "elapsed_ms": elapsed_ms,
                    "status": "ok",
                }
            )

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
        canvas._begin = None
        canvas._current = None

        for n in data.get("nodes", []):
            ntype = n.get("type", "begin")
            nid = n.get("id") or uuid.uuid4().hex[:8]
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
                comp = ToolComponent(
                    nid, name, tool_name=cfg.get("tool_name", "verifier"), position=pos
                )
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
            if src and tgt:
                canvas.add_edge(src, tgt, lbl)
                if (
                    src in canvas._nodes
                    and tgt in canvas._nodes
                    and canvas._nodes[src].next is None
                    and lbl == "default"
                ):
                    canvas._nodes[src].then(canvas._nodes[tgt])

        return canvas

    @classmethod
    def create_default_rag_canvas(cls, kb_id: str = "default") -> AgentCanvas:
        """创建预置的生产级 Hybrid + Knowledge Graph + 证据护栏 + 引用生成工作流。"""
        canvas = cls(
            name="Campus & Research Grounded RAG Workflow",
            description="Hybrid + Knowledge Graph retrieval -> Lexical Reranker -> Evidence Guardrail Switch -> Grounded Citation Generator",
            canvas_id="default-grounded-rag",
        )
        assert canvas._begin is not None
        canvas._begin.position = {"x": 60, "y": 180}
        ret = RetrievalComponent(
            "retrieve_1",
            "Hybrid + KG Retrieval",
            kb_id=kb_id,
            top_k=6,
            mode="graph_enhanced",
            position={"x": 240, "y": 180},
        )
        rerank = RerankComponent(
            "rerank_1",
            "Lexical Overlap Reranker",
            min_score=0.08,
            top_k=4,
            position={"x": 450, "y": 180},
        )
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
        gen = GenerateComponent(
            "generate_1",
            "Grounded Citation Generator",
            min_relevance_score=0.12,
            position={"x": 880, "y": 180},
        )
        end = EndComponent("end_1", "End", position={"x": 1080, "y": 180})

        canvas.add_component(ret)
        canvas.add_component(rerank)
        canvas.add_component(switch)
        canvas.add_component(gen)
        canvas.add_component(end)
        return canvas
