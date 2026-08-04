from __future__ import annotations

import re
import uuid
from collections import defaultdict
from typing import Optional

import numpy as np

from velorag.core.settings import GraphSettings
from velorag.core.types import Chunk, EntityNode, RelationEdge, SearchResult


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
                self._entity_to_chunks[entity.id].append(chunk.id)
                self._chunk_to_entities[chunk.id].append(entity.id)

            relations = self._extract_relations(chunk.text)
            for subj, rel, obj in relations:
                subj_entity = self._get_or_create_entity(subj, "concept")
                obj_entity = self._get_or_create_entity(obj, "concept")
                edge = RelationEdge(
                    id=uuid.uuid4().hex,
                    source_id=subj_entity.id,
                    target_id=obj_entity.id,
                    relation=rel,
                )
                self._relations[edge.id] = edge

    def _get_or_create_entity(self, name: str, entity_type: str = "concept") -> EntityNode:
        existing = self._name_index.get(name.lower())
        if existing:
            return self._entities[existing[0]]

        entity = EntityNode(
            id=uuid.uuid4().hex,
            name=name,
            entity_type=entity_type,
        )
        self._entities[entity.id] = entity
        self._name_index[name.lower()].append(entity.id)
        return entity

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        entities = []
        patterns = [
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', "named_entity"),
            (r'\b\d{4}-\d{2}-\d{2}\b', "date"),
            (r'\b\d+(?:\.\d+)?%\b', "percentage"),
        ]
        for pattern, etype in patterns:
            for match in re.finditer(pattern, text):
                entities.append((match.group(), etype))
        return list(set(entities))[:50]

    def _extract_relations(self, text: str) -> list[tuple[str, str, str]]:
        relations = []
        patterns = [
            (r'(\w+)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+)', "is_a"),
            (r'(\w+)\s+(?:contains|includes)\s+(\w+)', "contains"),
            (r'(\w+)\s+(?:causes|leads to|results in)\s+(\w+)', "causes"),
            (r'(\w+)\s+(?:depends on|relies on|requires)\s+(\w+)', "depends_on"),
            (r'(\w+)\s+(?:belongs to|part of)\s+(\w+)', "part_of"),
        ]
        for pattern, relation in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                subj, obj = match.group(1), match.group(2)
                if subj.lower() != obj.lower():
                    relations.append((subj, relation, obj))
        return relations[:20]

    def get_chunks_for_entity(self, entity_id: str, max_distance: int = 2) -> list[str]:
        visited = set()
        queue = [(entity_id, 0)]
        all_chunks = []

        while queue:
            current_id, dist = queue.pop(0)
            if current_id in visited or dist > max_distance:
                continue
            visited.add(current_id)

            chunks = self._entity_to_chunks.get(current_id, [])
            all_chunks.extend(chunks)

            for edge in self._relations.values():
                if edge.source_id == current_id and edge.target_id not in visited:
                    queue.append((edge.target_id, dist + 1))
                elif edge.target_id == current_id and edge.source_id not in visited:
                    queue.append((edge.source_id, dist + 1))

        return list(set(all_chunks))

    def find_relevant_entities(self, query: str) -> list[EntityNode]:
        query_lower = query.lower()
        matched = []
        for name, entity_ids in self._name_index.items():
            if name in query_lower:
                for eid in entity_ids:
                    matched.append(self._entities[eid])
        return matched[:10]

    def get_entity_count(self) -> int:
        return len(self._entities)

    def get_relation_count(self) -> int:
        return len(self._relations)
