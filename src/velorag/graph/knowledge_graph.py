from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Any, Optional

from velorag.core.settings import GraphSettings
from velorag.core.types import Chunk, EntityNode, RelationEdge


class KnowledgeGraph:
    def __init__(self, settings: GraphSettings | None = None):
        self.settings = settings or GraphSettings()
        self._entities: dict[str, EntityNode] = {}
        self._relations: dict[str, RelationEdge] = {}
        self._entity_to_chunks: dict[str, list[str]] = defaultdict(list)
        self._chunk_to_entities: dict[str, list[str]] = defaultdict(list)
        self._name_index: dict[str, list[str]] = defaultdict(list)

    def extract_from_chunks(self, chunks: list[Chunk]):
        for chunk in chunks:
            entities = self._extract_entities(chunk.text)
            for name, entity_type in entities:
                entity = self._get_or_create_entity(name, entity_type)
                if chunk.id not in self._entity_to_chunks[entity.id]:
                    self._entity_to_chunks[entity.id].append(chunk.id)
                if chunk.id not in entity.chunk_ids:
                    entity.chunk_ids.append(chunk.id)
                if entity.id not in self._chunk_to_entities[chunk.id]:
                    self._chunk_to_entities[chunk.id].append(entity.id)

            relations = self._extract_relations(chunk.text)
            for subj, rel, obj in relations:
                subj_entity = self._get_or_create_entity(subj, "concept")
                obj_entity = self._get_or_create_entity(obj, "concept")
                for ent in (subj_entity, obj_entity):
                    if chunk.id not in self._entity_to_chunks[ent.id]:
                        self._entity_to_chunks[ent.id].append(chunk.id)
                    if chunk.id not in ent.chunk_ids:
                        ent.chunk_ids.append(chunk.id)
                edge = RelationEdge(
                    id=uuid.uuid4().hex,
                    source_id=subj_entity.id,
                    target_id=obj_entity.id,
                    relation=rel,
                    chunk_id=chunk.id,
                )
                self._relations[edge.id] = edge

    def _get_or_create_entity(self, name: str, entity_type: str = "concept") -> EntityNode:
        clean_name = name.strip()
        existing = self._name_index.get(clean_name.lower())
        if existing:
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
            (r"([一-鿿]{2,8}(?:大学|学院|实验室|图书馆|教务处|系统|模型|算法|框架|助手|检索|图谱|评测))", "domain_entity"),
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
            (r"(\b[A-Za-z][A-Za-z0-9_-]+(?:\s+[A-Z][A-Za-z0-9_-]+)?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z0-9_-]+)", "is_a"),
            (r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:contains|includes|supports|uses)\s+([A-Za-z][A-Za-z0-9_-]+)", "contains"),
            (r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:causes|leads to|results in)\s+([A-Za-z][A-Za-z0-9_-]+)", "causes"),
            (r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:depends on|relies on|requires)\s+([A-Za-z][A-Za-z0-9_-]+)", "depends_on"),
            (r"(\b[A-Za-z][A-Za-z0-9_-]+)\s+(?:belongs to|part of)\s+([A-Za-z][A-Za-z0-9_-]+)", "part_of"),
        ]
        zh_patterns = [
            (r"([A-Za-z0-9_一-鿿]{2,12})(?:是|属于|作为)(?:一种|一个|一项)?([A-Za-z0-9_一-鿿]{2,12})", "is_a"),
            (r"([A-Za-z0-9_一-鿿]{2,12})(?:包含|包括|支持|集成|提供)([A-Za-z0-9_一-鿿]{2,12})", "contains"),
            (r"([A-Za-z0-9_一-鿿]{2,12})(?:依赖|基于|需要|采用)([A-Za-z0-9_一-鿿]{2,12})", "depends_on"),
            (r"([A-Za-z0-9_一-鿿]{2,12})(?:导致|引起|决定|提升)([A-Za-z0-9_一-鿿]{2,12})", "causes"),
        ]
        for pattern, relation in en_patterns + zh_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj, obj = match.group(1).strip(), match.group(2).strip()
                if subj.lower() != obj.lower() and len(subj) >= 2 and len(obj) >= 2:
                    relations.append((subj, relation, obj))
        return relations[:30]

    def get_chunks_with_distance(self, entity_id: str, max_distance: int = 2) -> dict[str, int]:
        """Return mapping of chunk_id -> minimum hop distance from entity_id."""
        visited: dict[str, int] = {}
        queue = [(entity_id, 0)]
        chunk_distances: dict[str, int] = {}

        while queue:
            current_id, dist = queue.pop(0)
            if current_id in visited and visited[current_id] <= dist:
                continue
            if dist > max_distance:
                continue
            visited[current_id] = dist

            for cid in self._entity_to_chunks.get(current_id, []):
                if cid not in chunk_distances or dist < chunk_distances[cid]:
                    chunk_distances[cid] = dist

            for edge in self._relations.values():
                if edge.source_id == current_id and dist + 1 <= max_distance:
                    queue.append((edge.target_id, dist + 1))
                elif edge.target_id == current_id and dist + 1 <= max_distance:
                    queue.append((edge.source_id, dist + 1))

        return chunk_distances

    def get_chunks_for_entity(self, entity_id: str, max_distance: int = 2) -> list[str]:
        return list(self.get_chunks_with_distance(entity_id, max_distance).keys())

    def find_relevant_entities(self, query: str) -> list[EntityNode]:
        query_lower = query.lower()
        matched: list[EntityNode] = []
        seen: set[str] = set()
        for name, entity_ids in self._name_index.items():
            if name in query_lower or query_lower in name:
                for eid in entity_ids:
                    if eid not in seen:
                        seen.add(eid)
                        matched.append(self._entities[eid])
        return matched[:15]

    def get_subgraph(self, query: str | None = None, max_hops: int = 2, limit: int = 50) -> dict[str, Any]:
        if query:
            seeds = self.find_relevant_entities(query)
            seed_ids = {e.id for e in seeds}
            nodes = {e.id: e for e in seeds}
            edges = []
            for edge in self._relations.values():
                if edge.source_id in seed_ids or edge.target_id in seed_ids:
                    edges.append(edge)
                    if edge.source_id in self._entities:
                        nodes[edge.source_id] = self._entities[edge.source_id]
                    if edge.target_id in self._entities:
                        nodes[edge.target_id] = self._entities[edge.target_id]
        else:
            node_list = list(self._entities.values())[:limit]
            nodes = {e.id: e for e in node_list}
            edges = [
                edge for edge in self._relations.values()
                if edge.source_id in nodes and edge.target_id in nodes
            ][: limit * 2]

        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "label": n.name,
                    "entity_type": n.entity_type,
                    "chunk_count": len(self._entity_to_chunks.get(n.id, [])),
                }
                for n in nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": self._entities[e.source_id].name if e.source_id in self._entities else e.source_id,
                    "source_id": e.source_id,
                    "source_name": self._entities[e.source_id].name if e.source_id in self._entities else "",
                    "target": self._entities[e.target_id].name if e.target_id in self._entities else e.target_id,
                    "target_id": e.target_id,
                    "target_name": self._entities[e.target_id].name if e.target_id in self._entities else "",
                    "relation": e.relation,
                    "weight": e.weight,
                }
                for e in edges
            ],
            "total_entities": len(self._entities),
            "total_relations": len(self._relations),
        }

    @property
    def node_count(self) -> int:
        return len(self._entities)

    @property
    def edge_count(self) -> int:
        return len(self._relations)

    def get_entity_count(self) -> int:
        return len(self._entities)

    def get_relation_count(self) -> int:
        return len(self._relations)
