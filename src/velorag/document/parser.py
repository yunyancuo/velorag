from __future__ import annotations

import io
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

from velorag.core.types import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, content: bytes, metadata: dict[str, Any] | None = None) -> Document:
        ...


class TextParser(BaseParser):
    def parse(self, content: bytes, metadata: dict[str, Any] | None = None) -> Document:
        text = content.decode("utf-8", errors="replace")
        return Document(
            id=uuid.uuid4().hex,
            content=text,
            metadata=dict(metadata) if metadata else {},
        )


class PDFParser(BaseParser):
    def parse(self, content: bytes, metadata: dict[str, Any] | None = None) -> Document:
        from PyPDF2 import PdfReader

        try:
            reader = PdfReader(io.BytesIO(content))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        except Exception as exc:
            raise ValueError(f"PDF 文件解析失败: {exc}") from exc

        return Document(
            id=uuid.uuid4().hex,
            content="\n\n".join(text_parts),
            metadata=dict(metadata) if metadata else {},
        )


class DocxParser(BaseParser):
    def parse(self, content: bytes, metadata: dict[str, Any] | None = None) -> Document:
        from docx import Document as DocxDoc

        try:
            doc = DocxDoc(io.BytesIO(content))
            text_parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            # 补充提取 Word 表格中的结构化文本，避免丢失表格关键信息
            for table in doc.tables:
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if cells:
                        text_parts.append(" | ".join(cells))
        except Exception as exc:
            raise ValueError(f"DOCX 文件解析失败: {exc}") from exc

        return Document(
            id=uuid.uuid4().hex,
            content="\n".join(text_parts),
            metadata=dict(metadata) if metadata else {},
        )


class MarkdownParser(BaseParser):
    def parse(self, content: bytes, metadata: dict[str, Any] | None = None) -> Document:
        text = content.decode("utf-8", errors="replace")
        cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        cleaned = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", cleaned)
        return Document(
            id=uuid.uuid4().hex,
            content=cleaned.strip(),
            metadata=dict(metadata) if metadata else {},
        )


PARSER_MAP: dict[str, type[BaseParser]] = {
    ".txt": TextParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".pdf": PDFParser,
    ".docx": DocxParser,
}
