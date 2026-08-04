from __future__ import annotations

import re
import uuid
from typing import Optional

from velorag.core.settings import ChunkSettings
from velorag.core.types import Chunk, ChunkStrategy, Document


class Chunker:
    def __init__(self, settings: ChunkSettings | None = None):
        self.settings = settings or ChunkSettings()

    def chunk(self, doc: Document) -> list[Chunk]:
        if self.settings.strategy == ChunkStrategy.RECURSIVE:
            return self._recursive_chunk(doc)
        elif self.settings.strategy == ChunkStrategy.SENTENCE:
            return self._sentence_chunk(doc)
        elif self.settings.strategy == ChunkStrategy.SEMANTIC:
            return self._semantic_chunk(doc)
        else:
            return self._fixed_chunk(doc)

    def _fixed_chunk(self, doc: Document) -> list[Chunk]:
        chunks = []
        text = doc.content
        size = self.settings.chunk_size
        overlap = self.settings.chunk_overlap
        step = max(1, size - overlap)

        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + size]
            if not chunk_text.strip():
                continue
            chunks.append(Chunk(
                id=uuid.uuid4().hex,
                text=chunk_text,
                doc_id=doc.id,
                chunk_index=i,
                metadata={**doc.metadata, "chunk_start": start},
            ))
        return chunks

    def _recursive_chunk(self, doc: Document) -> list[Chunk]:
        separators = self.settings.separators
        chunk_size = self.settings.chunk_size
        return self._split_text(doc, doc.content, separators, chunk_size)

    def _split_text(
        self, doc: Document, text: str, separators: list[str], chunk_size: int
    ) -> list[Chunk]:
        if not text.strip():
            return []

        sep = separators[0] if separators else " "
        splits = [s for s in text.split(sep) if s.strip()]
        chunks = []
        current = ""
        idx = 0

        for split in splits:
            if len(current) + len(split) + len(sep) <= chunk_size:
                current = current + sep + split if current else split
            else:
                if current:
                    chunks.append(self._make_chunk(doc, current.strip(), idx))
                    idx += 1
                if len(split) > chunk_size and len(separators) > 1:
                    sub_chunks = self._split_text(doc, split, separators[1:], chunk_size)
                    chunks.extend(sub_chunks)
                    idx += len(sub_chunks)
                else:
                    current = split

        if current.strip():
            chunks.append(self._make_chunk(doc, current.strip(), idx))

        return chunks

    def _sentence_chunk(self, doc: Document) -> list[Chunk]:
        sentences = re.split(r"(?<=[.。!?！？])\s+", doc.content)
        chunks = []
        current = ""
        idx = 0

        for sent in sentences:
            if len(current) + len(sent) <= self.settings.chunk_size:
                current = (current + " " + sent).strip() if current else sent
            else:
                if current:
                    chunks.append(self._make_chunk(doc, current, idx))
                    idx += 1
                current = sent

        if current.strip():
            chunks.append(self._make_chunk(doc, current.strip(), idx))

        return chunks

    def _semantic_chunk(self, doc: Document) -> list[Chunk]:
        paragraphs = [p.strip() for p in doc.content.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        idx = 0

        for para in paragraphs:
            if len(current) + len(para) <= self.settings.chunk_size:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(self._make_chunk(doc, current, idx))
                    idx += 1
                current = para

        if current.strip():
            chunks.append(self._make_chunk(doc, current.strip(), idx))

        return chunks

    def _make_chunk(self, doc: Document, text: str, idx: int) -> Chunk:
        return Chunk(
            id=uuid.uuid4().hex,
            text=text,
            doc_id=doc.id,
            chunk_index=idx,
            metadata={**doc.metadata},
        )
