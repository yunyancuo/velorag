from __future__ import annotations

import re
import uuid
from typing import Any, Sequence

from velorag.core.config import ChunkingConfig
from velorag.core.types import Chunk, ChunkingStrategy, Document


class Chunker:
    def __init__(self, settings: Any = None):
        self.settings = settings or ChunkingConfig()

    def _get_size_and_overlap(self) -> tuple[int, int]:
        size = max(1, int(getattr(self.settings, "chunk_size", 512)))
        raw_overlap = max(0, int(getattr(self.settings, "chunk_overlap", 64)))
        overlap = min(raw_overlap, size - 1) if size > 1 else 0
        return size, overlap

    def chunk_documents(self, docs: Sequence[Document]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    def chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text or doc.content
        if not text or not text.strip():
            return []

        strategy = getattr(self.settings, "strategy", ChunkingStrategy.RECURSIVE)
        if strategy == ChunkingStrategy.RECURSIVE:
            chunks = self._recursive_chunk(doc)
        elif strategy == ChunkingStrategy.SENTENCE:
            chunks = self._sentence_chunk(doc)
        elif strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._semantic_chunk(doc)
        else:
            chunks = self._fixed_chunk(doc)

        # 统一规范连续 chunk_index，避免递归切分产生重复或跳跃索引
        for idx, chunk in enumerate(chunks):
            chunk.chunk_index = idx
        return chunks

    def _fixed_chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        text = doc.text or doc.content
        size, overlap = self._get_size_and_overlap()
        step = max(1, size - overlap)

        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start : start + size]
            if not chunk_text.strip():
                continue
            chunks.append(
                Chunk(
                    id=uuid.uuid4().hex[:12],
                    text=chunk_text,
                    doc_id=doc.id,
                    chunk_index=i,
                    metadata={
                        **doc.metadata,
                        "source": doc.source or doc.metadata.get("source", ""),
                        "chunk_start": start,
                    },
                    start_char=start,
                    end_char=min(len(text), start + size),
                )
            )
            if start + size >= len(text):
                break
        return chunks

    def _recursive_chunk(self, doc: Document) -> list[Chunk]:
        separators = list(
            getattr(self.settings, "separators", ["\n\n", "\n", "。", ". ", " ", ""])
        )
        chunk_size, _ = self._get_size_and_overlap()
        text = doc.text or doc.content
        return self._split_text(doc, text, separators, chunk_size)

    def _split_text(
        self, doc: Document, text: str, separators: list[str], chunk_size: int
    ) -> list[Chunk]:
        if not text.strip():
            return []

        if len(text.strip()) <= chunk_size:
            return [self._make_chunk(doc, text.strip(), 0)]

        sep = separators[0] if separators else ""
        if sep == "":
            return self._fixed_chunk(
                Document(id=doc.id, text=text, metadata=doc.metadata, source=doc.source)
            )

        splits = [s for s in text.split(sep) if s.strip()]
        if len(splits) <= 1 and len(separators) > 1:
            return self._split_text(doc, text, separators[1:], chunk_size)

        chunks: list[Chunk] = []
        current = ""
        idx = 0

        for split in splits:
            candidate_len = len(current) + len(sep) + len(split) if current else len(split)
            if candidate_len <= chunk_size:
                current = current + sep + split if current else split
            else:
                if current.strip():
                    chunks.append(self._make_chunk(doc, current.strip(), idx))
                    idx += 1
                    current = ""

                if len(split) > chunk_size:
                    if len(separators) > 1:
                        sub_chunks = self._split_text(doc, split, separators[1:], chunk_size)
                    else:
                        sub_chunks = self._fixed_chunk(
                            Document(id=doc.id, text=split, metadata=doc.metadata, source=doc.source)
                        )
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
        chunks: list[Chunk] = []
        current = ""
        idx = 0
        chunk_size, _ = self._get_size_and_overlap()

        for sent in sentences:
            if len(current) + (1 if current else 0) + len(sent) <= chunk_size:
                current = (current + " " + sent).strip() if current else sent
            else:
                if current.strip():
                    chunks.append(self._make_chunk(doc, current.strip(), idx))
                    idx += 1
                    current = ""
                if len(sent) > chunk_size:
                    sub_chunks = self._fixed_chunk(
                        Document(id=doc.id, text=sent, metadata=doc.metadata, source=doc.source)
                    )
                    chunks.extend(sub_chunks)
                    idx += len(sub_chunks)
                else:
                    current = sent

        if current.strip():
            chunks.append(self._make_chunk(doc, current.strip(), idx))

        return chunks

    def _semantic_chunk(self, doc: Document) -> list[Chunk]:
        text = doc.text or doc.content
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        current = ""
        idx = 0
        chunk_size, _ = self._get_size_and_overlap()

        for para in paragraphs:
            if len(current) + (2 if current else 0) + len(para) <= chunk_size:
                current = current + "\n\n" + para if current else para
            else:
                if current.strip():
                    chunks.append(self._make_chunk(doc, current.strip(), idx))
                    idx += 1
                    current = ""
                if len(para) > chunk_size:
                    sub_chunks = self._sentence_chunk(
                        Document(id=doc.id, text=para, metadata=doc.metadata, source=doc.source)
                    )
                    chunks.extend(sub_chunks)
                    idx += len(sub_chunks)
                else:
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
