from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

from velorag.core.config import ChunkingConfig
from velorag.core.types import Chunk, ChunkingStrategy, Document


class Chunker:
    def __init__(self, settings: Any = None):
        self.settings = settings or ChunkingConfig()

    def chunk_documents(self, docs: Sequence[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    def chunk(self, doc: Document) -> list[Chunk]:
        strategy = getattr(self.settings, "strategy", ChunkingStrategy.RECURSIVE)
        if strategy == ChunkingStrategy.RECURSIVE:
            return self._recursive_chunk(doc)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunk(doc)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunk(doc)
        else:
            return self._fixed_chunk(doc)

    def _fixed_chunk(self, doc: Document) -> list[Chunk]:
        chunks = []
        text = doc.text or doc.content
        size = int(getattr(self.settings, "chunk_size", 512))
        overlap = int(getattr(self.settings, "chunk_overlap", 64))
        step = max(1, size - overlap)

        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + size]
            if not chunk_text.strip():
                continue
            chunks.append(Chunk(
                id=uuid.uuid4().hex[:12],
                text=chunk_text,
                doc_id=doc.id,
                chunk_index=i,
                metadata={**doc.metadata, "source": doc.source or doc.metadata.get("source", ""), "chunk_start": start},
                start_char=start,
                end_char=min(len(text), start + size),
            ))
        return chunks

    def _recursive_chunk(self, doc: Document) -> list[Chunk]:
        separators = getattr(self.settings, "separators", ["\n\n", "\n", "。", ". ", " ", ""])
        chunk_size = int(getattr(self.settings, "chunk_size", 512))
        text = doc.text or doc.content
        return self._split_text(doc, text, separators, chunk_size)

    def _split_text(
        self, doc: Document, text: str, separators: list[str], chunk_size: int
    ) -> list[Chunk]:
        if not text.strip():
            return []

        sep = separators[0] if separators else ""
        if sep == "":
            return self._fixed_chunk(
                Document(id=doc.id, text=text, metadata=doc.metadata, source=doc.source)
            )

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
        text = doc.text or doc.content
        sentences = [s.strip() for s in re.split(r"(?<=[.。!?！？])\s*", text) if s.strip()]
        chunks = []
        current = ""
        idx = 0
        chunk_size = int(getattr(self.settings, "chunk_size", 512))

        for sent in sentences:
            if len(current) + len(sent) <= chunk_size:
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
        text = doc.text or doc.content
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""
        idx = 0
        chunk_size = int(getattr(self.settings, "chunk_size", 512))

        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
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
            id=uuid.uuid4().hex[:12],
            text=text,
            doc_id=doc.id,
            chunk_index=idx,
            metadata={**doc.metadata, "source": doc.source or doc.metadata.get("source", "")},
        )
