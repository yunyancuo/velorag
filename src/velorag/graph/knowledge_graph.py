from __future__ import annotations

import re
import threading
import uuid
from collections import defaultdict, deque
from typing import Any, Optional

from velorag.core.settings import GraphSettings
from velorag.core.types import Chunk, EntityNode, RelationEdge


class KnowledgeGraph:
    """线程安全的双语实体关系知识图谱，支持 O(V+E) 邻接表多跳遍历与租户级子图隔离。"""

    def __init__(self, settings: GraphSettings | None = None):
        self.settings = settings or GraphSettings()
        self._entities: dict[str, EntityNode] = {}
        self._relations: dict[str, RelationEdge] = {}
        self._relation_keys: set[tuple[str, str, str, str]] = set()
        self._adj: dict[str, set[str]] = defaultdict(set)
        self._entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        self._chunk_to_entities: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, list[str]] = defaultdict(list)
        self._lock = threading.RLock()

    def _link_entity_chunk(self, entity: EntityNode, chunk_id: str) -> None:
        if chunk_id not in self._entity_to_chunks[entity.id]:
            self._entity_to_chunks[entity.id].append(chunk_id)
        if chunk_id not in entity.chunk_ids:
            entity.chunk_ids.append(chunk_id)
        if entity.id not in self._chunk_to_entities[chunk_id]:
            self._chunk_to_entities[chunk_id].append(entity.id)

    def extract_from_chunks(self, chunks: list[Chunk]):
        with self._lock:
            for chunk in chunks:
                entities = self._extract_entities(chunk.text)
                for name, entity_type in entities:
                    entity = self._get_or_create_entity(name, entity_type)
                    self._link_entity_chunk(entity, chunk.id)

                relations = self._extract_relations(chunk.text)
                for subj, rel, obj in relations:
                    subj_entity = self._get_or_create_entity(subj, "concept")
                    obj_entity = self._get_or_create_entity(obj, "concept")
                    self._link_entity_chunk(subj_entity, chunk.id)
                    self._link_entity_chunk(obj_entity, chunk.id)

                    rel_key = (subj_entity.id, obj_entity.id, rel, chunk.id)
                    if rel_key in self._relation_keys:
                        continue
                    self._relation_keys.add(rel_key)
                    edge = RelationEdge(
                        id=uuid.uuid4().hex,
                        source_id=subj_entity.id,
                        target_id=obj_entity.id,
                        relation=rel,
                        chunk_id=chunk.id,
                    )
                    self._relations[edge.id] = edge
                    self._adj[subj_entity.id].add(edge.id)
                    self._adj[obj_entity.id].add(edge.id)

    def remove_chunks(self, chunk_ids: set[str] | list[str]) -> None:
        """移除指定 chunk 关联并级联清理失效关系与孤立实体节点。"""
        remove_set = set(chunk_ids)
        if not remove_set:
            return
        with self._lock:
            # 1. 移除来源属于待删 chunk 的关系边
            edges_to_delete = [
                eid for eid, edge in self._relations.items() if edge.chunk_id in remove_set
            ]
            for eid in edges_to_delete:
                edge = self._relations.pop(eid)
                self._relation_keys.discard(
                    (edge.source_id, edge.target_id, edge.relation, edge.chunk_id)
                )
                self._adj[edge.source_id].discard(eid)
                self._adj[edge.target_id].discard(eid)

            # 2. 更新实体与 chunk 映射，清理不再关联任何 chunk 的孤立实体
            affected_entities: set[str] = set()
            for cid in remove_set:
                affected_entities.update(self._chunk_to_entities.pop(cid, []))

            for entity_id in affected_entities:
                remaining_chunks = [
                    cid for cid in self._entity_to_chunks.get(entity_id, []) if cid not in remove_set
                ]
                entity = self._entities.get(entity_id)
                if remaining_chunks:
                    self._entity_to_chunks[entity_id] = remaining_chunks
                    if entity is not None:
                        entity.chunk_ids = list(remaining_chunks)
                else:
                    self._entity_to_chunks.pop(entity_id, None)
                    self._adj.pop(entity_id, None)
                    removed_entity = self._entities.pop(entity_id, None)
                    if removed_entity is not None:
                        key = removed_entity.name.lower()
                        if key in self._name_index:
                            self._name_index[key] = [
                                eid for eid in self._name_index[key] if eid != entity_id
                            ]
                            if not self._name_index[key]:
                                del self._name_index[key]

    def _get_or_create_entity(self, name: str, entity_type: str = "concept") -> EntityNode:
        clean_name = name.strip()
        existing = self._name_index.get(clean_name.lower())
        if existing and existing[0] in self._entities:
            return self._entities[existing[0]]

        entity = EntityNode(
            id=uuid.uuid4().hex,
            name=clean_name,
            entity_type=entity_type,
        )
        self._entities[entity.id] = entity
        self._name_index[clean_name.lower()].append(entity.id)
        return entity

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        entities = []
        patterns = [
            (r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b", "named_entity"),
            (r"《([^》]{2,24})》", "document_or_title"),
            (r"「([^」]{2,20})」|“([^”]{2,20})”", "term"),
            (
                r"([\u4e00-\u9fff]{2,8}(?:大学|学院|实验室|图书馆|教务处|系统|模型|算法|框架|助手|检索|图谱|评测))",
                "domain_entity",
            ),
            (r"\b\d{4}-\d{2}-\d{2}\b", "date"),
            (r"\b\d+(?:\.\d+)?%\b", "percentage"),
        ]
        for pattern, etype in patterns:
            for match in re.finditer(pattern, text):
                val = next((g for g in match.groups() if g), match.group(0)).strip()
                if len(val) >= 2:
                    entities.append((val, etype))
        return list(set(entities))[:60]

    def _extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        relations = []
        en_patterns = [
            (
                r"(\b[A-Za-z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9_-]+)",
                "is_a",
            ),
            (
                r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:contains|includes|supports|uses)\s+([A-Za-z][A-Za-z0-9_-]+)",
                "contains",
            ),
            (
                r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:causes|leads to|results in)\s+([A-Za-z][A-Za-z0-9_-]+)",
                "causes",
            ),
            (
                r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:depends on|relies on|requires)\s+([A-Za-z][A-Za-z0-9_-]+)",
                "depends_on",
            ),
            (
                r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:belongs to|part of)\s+([A-Za-z][A-Za-z0-9_-]+)",
                "part_of",
            ),
        ]
        zh_patterns = [
            (
                r"([A-Za-z0-9_\u4e00-\u9fff]{2,12})(?:是|属于|作为)(?:一种|一个|一项)?([A-Za-z0-9_\u4e00-\u9fff]{2,12})",
                "is_a",
            ),
            (
                r"([A-Za-z0-9_\u4e00-\u9fff]{2,12})(?:包含|包括|支持|集成|提供)([A-Za-z0-9_\u4e00-\u9fff]{2,12})",
                "contains",
            ),
            (
                r"([A-Za-z0-9_\u4e00-\u9fff]{2,12})(?:依赖|基于|需要|采用)([A-Za-z0-9_\u4e00-\u9fff]{2,12})",
                "depends_on",
            ),
            (
                r"([A-Za-z0-9_\u4e00-\u9fff]{2,12})(?:导致|引起|决定|提升)([A-Za-z0-9_\u4e00-\u9fff]{2,12})",
                "causes",
            ),
        ]
        for pattern, relation in en_patterns + zh_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj, obj = match.group(1).strip(), match.group(2).strip()
                if subj.lower() != obj.lower() and len(subj) >= 2 and len(obj) >= 2:
                    relations.append((subj, relation, obj))
        return relations[:30]

    def get_chunks_with_distance(self, entity_id: str, max_distance: int = 2) -> dict[str, int]:
        """使用邻接表与 deque 执行 O(V+E) BFS，返回 chunk_id -> 最小跳数映射（天然免疫环路）。"""
        max_dist = max(0, int(max_distance))
        with self._lock:
            if entity_id not in self._entities:
                return {}

            visited: dict[str, int] = {entity_id: 0}
            queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
            chunk_distances: dict[str, int] = {}

            while queue:
                current_id, dist = queue.popleft()
                for cid in self._entity_to_chunks.get(current_id, []):
                    if cid not in chunk_distances or dist < chunk_distances[cid]:
                        chunk_distances[cid] = dist

                if dist >= max_dist:
                    continue

                for edge_id in self._adj.get(current_id, ()):
                    edge = self._relations.get(edge_id)
                    if edge is None:
                        continue
                    neighbor_id = (
                        edge.target_id if edge.source_id == current_id else edge.source_id
                    )
                    if neighbor_id not in visited and neighbor_id in self._entities:
                        visited[neighbor_id] = dist + 1
                        queue.append((neighbor_id, dist + 1))

            return chunk_distances

    def get_chunks_for_entity(self, entity_id: str, max_distance: int = 2) -> list[str]:
        return list(self.get_chunks_with_distance(entity_id, max_distance).keys())

    def find_relevant_entities(
        self, query: str, allowed_chunk_ids: Optional[set[str]] = None
    ) -> list[EntityNode]:
        """检索与查询相关的实体节点，拒绝空查询误匹配全图，并支持租户级 chunk 过滤。"""
        if not query or not query.strip():
            return []

        query_lower = query.strip().lower()
        with self._lock:
            matched: list[EntityNode] = []
            seen: set[str] = set()
            # 优先匹配名称较长的领域实体，提高召回精确度
            for name in sorted(self._name_index.keys(), key=len, reverse=True):
                if name in query_lower or (len(query_lower) >= 2 and query_lower in name):
                    for eid in self._name_index[name]:
                        if eid in seen or eid not in self._entities:
                            continue
                        if allowed_chunk_ids is not None:
                            entity_chunks = self._entity_to_chunks.get(eid, [])
                            if not any(cid in allowed_chunk_ids for cid in entity_chunks):
                                continue
                        seen.add(eid)
                        matched.append(self._entities[eid])
                        if len(matched) >= 15:
                            return matched
            return matched

    def get_subgraph(
        self,
        query: str | None = None,
        max_hops: int = 2,
        limit: int = 50,
        allowed_chunk_ids: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """按多跳 BFS 与严格数量上限导出知识图谱子图，支持按 allowed_chunk_ids 隔离多租户数据。"""
        safe_limit = max(1, min(int(limit), 500))
        safe_hops = max(0, min(int(max_hops), 5))

        with self._lock:
            def _entity_allowed(eid: str) -> bool:
                if allowed_chunk_ids is None:
                    return True
                return any(cid in allowed_chunk_ids for cid in self._entity_to_chunks.get(eid, []))

            def _edge_allowed(edge: RelationEdge) -> bool:
                if allowed_chunk_ids is None:
                    return True
                return edge.chunk_id in allowed_chunk_ids

            nodes: dict[str, EntityNode] = {}
            edges: list[RelationEdge] = []
            seen_edges: set[str] = set()

            if query and query.strip():
                seeds = self.find_relevant_entities(query, allowed_chunk_ids=allowed_chunk_ids)
                queue: deque[tuple[str, int]] = deque()
                visited: set[str] = set()

                for seed in seeds:
                    if len(nodes) >= safe_limit:
                        break
                    nodes[seed.id] = seed
                    visited.add(seed.id)
                    queue.append((seed.id, 0))

                while queue:
                    current_id, hop = queue.popleft()
                    if hop >= safe_hops:
                        continue
                    for edge_id in self._adj.get(current_id, ()):
                        edge = self._relations.get(edge_id)
                        if edge is None or not _edge_allowed(edge):
                            continue
                        neighbor_id = (
                            edge.target_id if edge.source_id == current_id else edge.source_id
                        )
                        if neighbor_id not in self._entities or not _entity_allowed(neighbor_id):
                            continue
                        if neighbor_id not in nodes and len(nodes) >= safe_limit:
                            continue
                        if neighbor_id not in nodes:
                            nodes[neighbor_id] = self._entities[neighbor_id]
                        if edge_id not in seen_edges and len(edges) < safe_limit * 2:
                            seen_edges.add(edge_id)
                            edges.append(edge)
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append((neighbor_id, hop + 1))
            else:
                for entity in self._entities.values():
                    if not _entity_allowed(entity.id):
                        continue
                    nodes[entity.id] = entity
                    if len(nodes) >= safe_limit:
                        break

                for edge in self._relations.values():
                    if not _edge_allowed(edge):
                        continue
                    if edge.source_id in nodes and edge.target_id in nodes:
                        edges.append(edge)
                        if len(edges) >= safe_limit * 2:
                            break

            total_entities = (
                sum(1 for eid in self._entities if _entity_allowed(eid))
                if allowed_chunk_ids is not None
                else len(self._entities)
            )
            total_relations = (
                sum(1 for edge in self._relations.values() if _edge_allowed(edge))
                if allowed_chunk_ids is not None
                else len(self._relations)
            )

            return {
                "nodes": [
                    {
                        "id": n.id,
                        "name": n.name,
                        "label": n.name,
                        "entity_type": n.entity_type,
                        "chunk_count": (
                            sum(
                                1
                                for cid in self._entity_to_chunks.get(n.id, [])
                                if allowed_chunk_ids is None or cid in allowed_chunk_ids
                            )
                        ),
                    }
                    for n in nodes.values()
                ],
                "edges": [
                    {
                        "id": e.id,
                        "source": (
                            self._entities[e.source_id].name
                            if e.source_id in self._entities
                            else e.source_id
                        ),
                        "source_id": e.source_id,
                        "source_name": (
                            self._entities[e.source_id].name
                            if e.source_id in self._entities
                            else ""
                        ),
                        "target": (
                            self._entities[e.target_id].name
                            if e.target_id in self._entities
                            else e.target_id
                        ),
                        "target_id": e.target_id,
                        "target_name": (
                            self._entities[e.target_id].name
                            if e.target_id in self._entities
                            else ""
                        ),
                        "relation": e.relation,
                        "weight": e.weight,
                    }
                    for e in edges
                ],
                "total_entities": total_entities,
                "total_relations": total_relations,
            }

    @property
    def node_count(self) -> int:
        with self._lock:
            return len(self._entities)

    @property
    def edge_count(self) -> int:
        with self._lock:
            return len(self._relations)

    def get_entity_count(self) -> int:
        return self.node_count

    def get_relation_count(self) -> int:
        return self.edge_count
