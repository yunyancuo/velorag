from __future__ import annotations

import ast
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

MAX_CODE_LENGTH = 2000
MAX_AST_NODES = 200
MAX_EVAL_STEPS = 500
MAX_COLLECTION_ITER = 200

_SAFE_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}

_SAFE_METHODS: dict[type, frozenset[str]] = {
    dict: frozenset({"get", "keys", "values", "items"}),
    str: frozenset(
        {
            "strip",
            "lstrip",
            "rstrip",
            "lower",
            "upper",
            "split",
            "startswith",
            "endswith",
            "replace",
            "join",
        }
    ),
    list: frozenset({"count", "index"}),
    tuple: frozenset({"count", "index"}),
}

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}

_CMP_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}

_ALLOWED_AST_TYPES = (
    ast.Module,
    ast.Assign,
    ast.Expr,
    ast.If,
    ast.IfExp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.Subscript,
    ast.Slice,
    ast.Attribute,
    ast.Call,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.ListComp,
    ast.comprehension,
    *tuple(_BIN_OPS.keys()),
    *tuple(_UNARY_OPS.keys()),
    *tuple(_CMP_OPS.keys()),
)


def validate_safe_code(code: str) -> ast.Module:
    """静态校验 CodeComponent 代码 AST，彻底阻断 import、dunder 反射逃逸与死循环。"""
    if not isinstance(code, str):
        raise ValueError("代码必须为字符串")
    if len(code) > MAX_CODE_LENGTH:
        raise ValueError(f"代码长度超过上限 ({MAX_CODE_LENGTH} 字符)")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"代码语法错误: {exc.msg}") from exc

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > MAX_AST_NODES:
            raise ValueError(f"代码 AST 节点数超过安全上限 ({MAX_AST_NODES})")
        if not isinstance(node, _ALLOWED_AST_TYPES):
            raise ValueError(f"禁止使用的语法结构: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"禁止访问内部或私有属性: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError(f"禁止访问内部标识符: {node.id}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError(f"禁止调用函数: {node.func.id}")
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("_"):
                raise ValueError(f"禁止调用内部方法: {node.func.attr}")
    return tree


class SafeCodeEvaluator:
    """基于白名单 AST 的受控表达式求值器，替代不安全的 exec() 以消除 RCE 漏洞。"""

    def __init__(self, env: dict[str, Any]):
        self._env = env
        self._steps = 0

    def _tick(self) -> None:
        self._steps += 1
        if self._steps > MAX_EVAL_STEPS:
            raise ValueError(f"代码执行步数超过安全上限 ({MAX_EVAL_STEPS})")

    def run(self, code: str) -> dict[str, Any]:
        tree = validate_safe_code(code)
        for stmt in tree.body:
            self._exec_stmt(stmt)
        return self._env

    def _exec_stmt(self, stmt: ast.stmt) -> None:
        self._tick()
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError("仅支持对单个简单变量名赋值")
            target_name = stmt.targets[0].id
            if target_name.startswith("_"):
                raise ValueError(f"禁止赋值给内部标识符: {target_name}")
            self._env[target_name] = self._eval_expr(stmt.value)
        elif isinstance(stmt, ast.If):
            branch = stmt.body if bool(self._eval_expr(stmt.test)) else stmt.orelse
            for child in branch:
                self._exec_stmt(child)
        elif isinstance(stmt, ast.Expr):
            self._eval_expr(stmt.value)
        else:
            raise ValueError(f"不支持的语句类型: {type(stmt).__name__}")

    def _eval_expr(self, node: ast.expr) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._env:
                return self._env[node.id]
            if node.id in _SAFE_FUNCTIONS:
                return _SAFE_FUNCTIONS[node.id]
            raise ValueError(f"未定义的变量: {node.id}")
        if isinstance(node, ast.List):
            return [self._eval_expr(elt) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_expr(elt) for elt in node.elts)
        if isinstance(node, ast.Set):
            return {self._eval_expr(elt) for elt in node.elts}
        if isinstance(node, ast.Dict):
            result: dict[Any, Any] = {}
            for k, v in zip(node.keys, node.values):
                if k is None:
                    raise ValueError("不支持字典解包语法")
                result[self._eval_expr(k)] = self._eval_expr(v)
            return result
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"不支持的一元运算符: {type(node.op).__name__}")
            return op_fn(self._eval_expr(node.operand))
        if isinstance(node, ast.BinOp):
            op_fn = _BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"不支持的二元运算符: {type(node.op).__name__}")
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            # 防止字符串/列表超大乘法导致内存 DoS
            if isinstance(node.op, ast.Mult) and isinstance(left, (str, list, tuple)):
                if isinstance(right, int) and right > 1000:
                    raise ValueError("序列重复倍数超过安全上限")
            return op_fn(left, right)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                val: Any = True
                for item in node.values:
                    val = self._eval_expr(item)
                    if not val:
                        return val
                return val
            if isinstance(node.op, ast.Or):
                val = False
                for item in node.values:
                    val = self._eval_expr(item)
                    if val:
                        return val
                return val
        if isinstance(node, ast.Compare):
            left = self._eval_expr(node.left)
            for op, comp in zip(node.ops, node.comparators):
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"不支持的比较运算符: {type(op).__name__}")
                right = self._eval_expr(comp)
                if not op_fn(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return (
                self._eval_expr(node.body)
                if bool(self._eval_expr(node.test))
                else self._eval_expr(node.orelse)
            )
        if isinstance(node, ast.Subscript):
            value = self._eval_expr(node.value)
            if isinstance(node.slice, ast.Slice):
                lower = self._eval_expr(node.slice.lower) if node.slice.lower else None
                upper = self._eval_expr(node.slice.upper) if node.slice.upper else None
                step = self._eval_expr(node.slice.step) if node.slice.step else None
                return value[slice(lower, upper, step)]
            return value[self._eval_expr(node.slice)]
        if isinstance(node, ast.ListComp):
            if len(node.generators) != 1:
                raise ValueError("仅支持单层列表推导式")
            gen = node.generators[0]
            if not isinstance(gen.target, ast.Name) or gen.target.id.startswith("_"):
                raise ValueError("列表推导式仅支持简单循环变量")
            iterable = list(self._eval_expr(gen.iter))
            if len(iterable) > MAX_COLLECTION_ITER:
                raise ValueError(f"列表推导迭代长度超过安全上限 ({MAX_COLLECTION_ITER})")
            saved_present = gen.target.id in self._env
            saved_value = self._env.get(gen.target.id)
            out_list: list[Any] = []
            try:
                for item in iterable:
                    self._tick()
                    self._env[gen.target.id] = item
                    if all(bool(self._eval_expr(cond)) for cond in gen.ifs):
                        out_list.append(self._eval_expr(node.elt))
            finally:
                if saved_present:
                    self._env[gen.target.id] = saved_value
                else:
                    self._env.pop(gen.target.id, None)
            return out_list
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("安全求值器暂不支持关键字参数调用")
            args = [self._eval_expr(a) for a in node.args]
            if isinstance(node.func, ast.Name):
                fn = _SAFE_FUNCTIONS.get(node.func.id)
                if fn is None:
                    raise ValueError(f"禁止调用函数: {node.func.id}")
                return fn(*args)
            if isinstance(node.func, ast.Attribute):
                owner = self._eval_expr(node.func.value)
                attr_name = node.func.attr
                allowed_for_type = _SAFE_METHODS.get(type(owner), frozenset())
                if attr_name not in allowed_for_type:
                    raise ValueError(
                        f"禁止在 {type(owner).__name__} 上调用方法: {attr_name}"
                    )
                return getattr(owner, attr_name)(*args)
            raise ValueError("不支持的调用目标")
        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


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

        filters: dict[str, Any] = {}
        param_filters = context.params.get("filters")
        if isinstance(param_filters, dict):
            filters.update(param_filters)

        # 显式透传 kb_id 作用域过滤条件（调用参数优先级高于组件默认配置）
        effective_kb_id = context.params.get("kb_id") or self.config.get("kb_id")
        if effective_kb_id and "kb_id" not in filters:
            filters["kb_id"] = str(effective_kb_id)

        results = rag.search(
            query=context.query,
            top_k=int(self.config.get("top_k", 5)),
            mode=str(self.config.get("mode", "hybrid")),
            use_graph=bool(self.config.get("use_graph", True)),
            filters=filters or None,
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
        try:
            if op == ">=":
                return float(val) >= float(thresh)
            if op == ">":
                return float(val) > float(thresh)
            if op == "<=":
                return float(val) <= float(thresh)
            if op == "<":
                return float(val) < float(thresh)
        except (TypeError, ValueError):
            return False
        if op == "==":
            return val == thresh
        if op == "!=":
            return val != thresh
        if op == "in":
            return isinstance(thresh, (list, tuple, set, str)) and val in thresh
        if op == "not_in":
            return isinstance(thresh, (list, tuple, set, str)) and val not in thresh
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
                "variables": dict(context.variables),
                "output": {},
            }
            evaluator = SafeCodeEvaluator(local_env)
            evaluator.run(str(self.config.get("code", "")))
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
        from velorag.core.types import Chunk as CoreChunk
        from velorag.core.types import GenerationRequest, SearchResult
        from velorag.generation.llm import LLMGenerator

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
