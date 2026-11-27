from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_DOCUMENT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".pdf", ".docx"})

# 仅允许安全文件名字符，避免路径遍历、空字节和控制字符注入
_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f<>:\"|?*]")


def sanitize_filename(filename: str) -> str:
    """校验并清洗上传文件名，彻底拦截路径遍历与非法字符。"""
    if not filename or not isinstance(filename, str):
        raise ValueError("文件名不能为空")

    raw = filename.strip()
    if not raw or raw in (".", ".."):
        raise ValueError("非法文件名")

    # 显式拒绝包含目录分隔符或上级目录跳转的输入
    if "/" in raw or "\\" in raw or ".." in Path(raw).parts:
        raise ValueError(f"检测到非法路径字符: {filename!r}")

    if _UNSAFE_FILENAME_CHARS.search(raw):
        raise ValueError(f"文件名包含非法控制字符或保留字符: {filename!r}")

    clean = Path(raw).name
    if not clean or clean in (".", ".."):
        raise ValueError("非法文件名")
    if len(clean) > 255:
        raise ValueError("文件名长度不能超过 255 个字符")

    return clean


def validate_document_extension(filename: str) -> str:
    """校验文档扩展名是否在安全白名单内并返回小写扩展名。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_DOCUMENT_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_DOCUMENT_EXTENSIONS))
        raise ValueError(f"不支持的文件类型: {suffix or '(无扩展名)'}，仅允许: {allowed}")
    return suffix


def resolve_safe_path(base_dir: Path, relative_name: str) -> Path:
    """将目标路径解析并强约束在 base_dir 目录之内，防止路径穿越和符号链接逃逸。"""
    base_resolved = base_dir.expanduser().resolve()
    target_resolved = (base_resolved / relative_name).resolve()
    try:
        target_resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"目标路径越界，已拒绝访问: {relative_name!r}") from exc
    return target_resolved


def validate_http_url(url: str) -> str:
    """校验外部 API 基地址仅使用 http/https 协议且具备合法主机名，防范非法 scheme SSRF。"""
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"非法的 HTTP(S) URL: {url!r}")
    return url.strip()
