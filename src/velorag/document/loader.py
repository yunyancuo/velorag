from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import aiofiles

from velorag.core.types import Document, Chunk
from velorag.core.settings import ChunkSettings
from velorag.document.parser import BaseParser, PARSER_MAP
from velorag.document.chunker import Chunker


class DocumentLoader:
    def __init__(self, chunk_settings: ChunkSettings | None = None):
        self.chunker = Chunker(chunk_settings)

    async def load_file(self, file_path: str | Path) -> list[Chunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        parser_cls = PARSER_MAP.get(suffix)
        if parser_cls is None:
            parser_cls = PARSER_MAP[".txt"]

        async with aiofiles.open(path, "rb") as f:
            content = await f.read()

        parser: BaseParser = parser_cls()
        doc = parser.parse(content, metadata={"source": str(path), "file_type": suffix})
        return self.chunker.chunk(doc)

    def load_file_sync(self, file_path: str | Path) -> list[Chunk]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        parser_cls = PARSER_MAP.get(suffix)
        if parser_cls is None:
            parser_cls = PARSER_MAP[".txt"]

        content = path.read_bytes()
        parser: BaseParser = parser_cls()
        doc = parser.parse(content, metadata={"source": str(path), "file_type": suffix})
        return self.chunker.chunk(doc)

    async def load_directory(self, dir_path: str | Path, recursive: bool = True) -> list[Chunk]:
        path = Path(dir_path)
        pattern = "**/*" if recursive else "*"
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower() in PARSER_MAP
        ]
        tasks = [self.load_file(f) for f in files]
        results = await asyncio.gather(*tasks)
        all_chunks = []
        for chunks in results:
            all_chunks.extend(chunks)
        return all_chunks
