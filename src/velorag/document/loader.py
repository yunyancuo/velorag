from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from velorag.core.types import Chunk, Document
from velorag.document.chunker import Chunker
from velorag.document.parser import BaseParser, PARSER_MAP


class DocumentLoader:
    def __init__(self, chunk_settings: Any = None):
        self.chunker = Chunker(chunk_settings)

    def load(self, path: str | Path) -> list[Document]:
        """Synchronously load a single file or directory of files into Document objects."""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if target.is_file():
            return [self._parse_file_sync(target)]

        docs: list[Document] = []
        for f in sorted(target.rglob("*")):
            if f.is_file() and f.suffix.lower() in PARSER_MAP:
                docs.append(self._parse_file_sync(f))
        return docs

    def _parse_file_sync(self, path: Path) -> Document:
        suffix = path.suffix.lower()
        parser_cls = PARSER_MAP.get(suffix, PARSER_MAP[".txt"])
        content = path.read_bytes()
        parser: BaseParser = parser_cls()
        doc = parser.parse(content, metadata={"source": path.name, "file_path": str(path), "file_type": suffix})
        doc.source = path.name
        return doc

    async def load_file(self, file_path: str | Path) -> list[Chunk]:
        doc = self._parse_file_sync(Path(file_path))
        return self.chunker.chunk(doc)

    def load_file_sync(self, file_path: str | Path) -> list[Chunk]:
        doc = self._parse_file_sync(Path(file_path))
        return self.chunker.chunk(doc)

    async def load_directory(self, dir_path: str | Path, recursive: bool = True) -> list[Chunk]:
        docs = self.load(dir_path)
        return self.chunker.chunk_documents(docs)
