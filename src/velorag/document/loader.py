from __future__ import annotations

from pathlib import Path
from typing import Any

from velorag.core.types import Chunk, Document
from velorag.document.chunker import Chunker
from velorag.document.parser import PARSER_MAP, BaseParser

DEFAULT_MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


class DocumentLoader:
    def __init__(
        self,
        chunk_settings: Any = None,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    ):
        self.chunker = Chunker(chunk_settings)
        self.max_file_size_bytes = max(1, int(max_file_size_bytes))

    def load(
        self,
        path: str | Path,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """同步加载单个文件或目录下的支持格式文档。"""
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if target.is_file():
            return [self._parse_file_sync(target, metadata=metadata)]

        iterator = target.rglob("*") if recursive else target.glob("*")
        docs: list[Document] = []
        for f in sorted(iterator):
            if f.is_file() and f.suffix.lower() in PARSER_MAP:
                docs.append(self._parse_file_sync(f, metadata=metadata))
        return docs

    def _parse_file_sync(
        self, path: Path, metadata: dict[str, Any] | None = None
    ) -> Document:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise ValueError(
                f"文件 {path.name} 体积 ({file_size} bytes) 超过上限 ({self.max_file_size_bytes} bytes)"
            )

        suffix = path.suffix.lower()
        parser_cls = PARSER_MAP.get(suffix, PARSER_MAP[".txt"])
        content = path.read_bytes()
        parser: BaseParser = parser_cls()
        merged_meta = {
            "source": path.name,
            "file_path": str(path),
            "file_type": suffix,
            **(metadata or {}),
        }
        doc = parser.parse(content, metadata=merged_meta)
        doc.source = str(merged_meta.get("source") or path.name)
        if metadata and "doc_id" in metadata and metadata["doc_id"]:
            doc.id = str(metadata["doc_id"])
        return doc

    async def load_file(
        self, file_path: str | Path, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        doc = self._parse_file_sync(Path(file_path), metadata=metadata)
        return self.chunker.chunk(doc)

    def load_file_sync(
        self, file_path: str | Path, metadata: dict[str, Any] | None = None
    ) -> list[Chunk]:
        doc = self._parse_file_sync(Path(file_path), metadata=metadata)
        return self.chunker.chunk(doc)

    async def load_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        docs = self.load(dir_path, recursive=recursive, metadata=metadata)
        return self.chunker.chunk_documents(docs)
