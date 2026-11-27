from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from velorag.auth.dependencies import get_current_user
from velorag.core.database import get_db
from velorag.core.security import (
    resolve_safe_path,
    sanitize_filename,
    validate_document_extension,
)
from velorag.core.settings import get_settings
from velorag.db.models import User
from velorag.knowledge.access import (
    DEFAULT_KB_ID,
    authorize_kb_access,
    ensure_default_kb,
)
from velorag.knowledge.models import DocumentModel, DocumentStatus, KnowledgeBaseModel
from velorag.knowledge.schemas import (
    DocumentResponse,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
)
from velorag.server.state import get_default_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_UPLOAD_READ_CHUNK_SIZE = 1024 * 64  # 64 KiB


class DirectTextCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=2_000_000)


def _max_upload_bytes() -> int:
    return get_settings().server.max_upload_mb * 1024 * 1024


def _safe_kb_dir(kb_id: str) -> Path:
    safe_kb = sanitize_filename(kb_id)
    kb_dir = resolve_safe_path(UPLOAD_DIR, safe_kb)
    kb_dir.mkdir(parents=True, exist_ok=True)
    return kb_dir


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    kb: KnowledgeBaseCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    data = kb.model_dump()
    # 已登录用户创建的知识库默认私有并绑定所有者；匿名创建保持向后兼容为公开演示库
    if data["is_public"] is None:
        data["is_public"] = user is None

    db_kb = KnowledgeBaseModel(
        **data,
        owner_id=user.id if user is not None else None,
    )
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)
    return db_kb


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_default_kb(db)
    query = db.query(KnowledgeBaseModel)
    if user is None:
        query = query.filter(KnowledgeBaseModel.is_public.is_(True))
    elif not user.is_superuser:
        query = query.filter(
            KnowledgeBaseModel.is_public.is_(True)
            | (KnowledgeBaseModel.owner_id == user.id)
        )
    return query.order_by(KnowledgeBaseModel.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(
    kb_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kb = authorize_kb_access(db, kb_id, user, write=False)
    assert kb is not None
    return kb


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_kb(
    kb_id: str,
    kb_update: KnowledgeBaseUpdate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kb = authorize_kb_access(db, kb_id, user, write=True)
    assert kb is not None

    update_data = kb_update.model_dump(exclude_unset=True)
    new_size = update_data.get("chunk_size", kb.chunk_size)
    new_overlap = update_data.get("chunk_overlap", kb.chunk_overlap)
    if new_overlap >= new_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="chunk_overlap must be strictly less than chunk_size",
        )

    for key, value in update_data.items():
        setattr(kb, key, value)

    db.commit()
    db.refresh(kb)
    return kb


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if kb_id == DEFAULT_KB_ID:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete the built-in default knowledge base",
        )
    kb = authorize_kb_access(db, kb_id, user, write=True)
    assert kb is not None

    # 先级联清理关联该知识库的聊天会话（及其消息），避免外键约束冲突
    from velorag.chat.models import ChatSessionModel

    for chat_session in (
        db.query(ChatSessionModel).filter(ChatSessionModel.kb_id == kb_id).all()
    ):
        db.delete(chat_session)

    db.delete(kb)
    db.commit()

    # 同步清理内存检索管线、知识图谱与本地上传目录，防止残留幽灵数据
    rag = get_default_pipeline()
    rag.delete_by_kb(kb_id)
    try:
        kb_dir = resolve_safe_path(UPLOAD_DIR, sanitize_filename(kb_id))
        if kb_dir.exists():
            shutil.rmtree(kb_dir, ignore_errors=True)
    except ValueError:
        pass

    return {"status": "deleted"}


@router.post("/{kb_id}/documents", response_model=DocumentResponse)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authorize_kb_access(db, kb_id, user, write=True)

    try:
        safe_filename = sanitize_filename(file.filename or "")
        validate_document_extension(safe_filename)
        kb_dir = _safe_kb_dir(kb_id)
        file_path = resolve_safe_path(kb_dir, safe_filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    max_bytes = _max_upload_bytes()
    total_written = 0
    try:
        with file_path.open("wb") as buffer:
            while True:
                chunk_bytes = await file.read(_UPLOAD_READ_CHUNK_SIZE)
                if not chunk_bytes:
                    break
                total_written += len(chunk_bytes)
                if total_written > max_bytes:
                    buffer.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {get_settings().server.max_upload_mb} MB",
                    )
                buffer.write(chunk_bytes)
    finally:
        await file.close()

    if total_written == 0:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    doc = DocumentModel(
        kb_id=kb_id,
        filename=safe_filename,
        file_path=str(file_path),
        file_size=total_written,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # 未启用 Redis/Celery 时使用零依赖同步解析入库回退方案
    use_celery = os.getenv("VELORAG_USE_CELERY", "0") == "1"
    if use_celery:
        try:
            from velorag.tasks.ingestion import process_document

            process_document.delay(doc.id, kb_id, str(file_path))
            return doc
        except Exception:
            logger.warning("Celery 队列不可用，自动回退至同步文档入库模式")

    try:
        doc.status = DocumentStatus.PARSING
        db.commit()
        rag = get_default_pipeline()
        # 显式注入 kb_id 与 doc_id 元数据，确保多租户知识库严格隔离
        chunk_count = rag.ingest(
            str(file_path),
            metadata={"kb_id": kb_id, "doc_id": doc.id, "source": safe_filename},
        )
        doc.chunk_count = chunk_count
        doc.status = DocumentStatus.INDEXED
        db.commit()
        db.refresh(doc)
    except Exception as exc:
        logger.exception("文档解析入库失败: doc_id=%s kb_id=%s", doc.id, kb_id)
        doc.status = DocumentStatus.FAILED
        doc.error_msg = f"Document parsing failed: {type(exc).__name__}"
        db.commit()
        db.refresh(doc)

    return doc


@router.post("/{kb_id}/texts", response_model=DocumentResponse)
def ingest_raw_text_document(
    kb_id: str,
    payload: DirectTextCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authorize_kb_access(db, kb_id, user, write=True)

    try:
        safe_filename = sanitize_filename(payload.filename)
        if Path(safe_filename).suffix:
            validate_document_extension(safe_filename)
        else:
            safe_filename = f"{safe_filename}.txt"
        kb_dir = _safe_kb_dir(kb_id)
        file_path = resolve_safe_path(kb_dir, safe_filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from None

    encoded = payload.content.encode("utf-8")
    if len(encoded) > _max_upload_bytes():
        raise HTTPException(
            status_code=413,
            detail=f"Content exceeds maximum allowed size of {get_settings().server.max_upload_mb} MB",
        )

    file_path.write_bytes(encoded)

    doc = DocumentModel(
        kb_id=kb_id,
        filename=safe_filename,
        file_path=str(file_path),
        file_size=len(encoded),
        chunk_count=0,
        status=DocumentStatus.INDEXING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    rag = get_default_pipeline()
    chunk_count = rag.ingest_text(
        text=payload.content,
        doc_id=doc.id,
        source=safe_filename,
        metadata={"kb_id": kb_id, "title": safe_filename},
    )
    doc.chunk_count = chunk_count
    doc.status = DocumentStatus.INDEXED
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
def list_documents(
    kb_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authorize_kb_access(db, kb_id, user, write=False)
    docs = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).all()
    if not docs and kb_id == DEFAULT_KB_ID:
        from velorag.eval.dataset import get_builtin_campus_benchmark

        builtin_docs, _ = get_builtin_campus_benchmark()
        seeded = []
        for bdoc in builtin_docs:
            dm = DocumentModel(
                id=bdoc.doc_id,
                kb_id=DEFAULT_KB_ID,
                filename=bdoc.metadata.get("source", f"{bdoc.doc_id}.md"),
                file_path=f"builtin://{bdoc.doc_id}",
                file_size=len(bdoc.text.encode("utf-8")),
                chunk_count=1,
                status=DocumentStatus.INDEXED,
            )
            db.add(dm)
            seeded.append(dm)
        db.commit()
        return seeded
    return docs
