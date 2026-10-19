import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from velorag.core.database import get_db
from velorag.knowledge.models import KnowledgeBaseModel, DocumentModel, DocumentStatus
from velorag.knowledge.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    DocumentResponse,
)
from velorag.server.state import get_default_pipeline

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base"])

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class DirectTextCreate(BaseModel):
    filename: str
    content: str


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(kb: KnowledgeBaseCreate, db: Session = Depends(get_db)):
    db_kb = KnowledgeBaseModel(**kb.model_dump())
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)
    return db_kb


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    kbs = db.query(KnowledgeBaseModel).offset(skip).limit(limit).all()
    if not kbs:
        # Auto-seed a default Campus & AI Lab KB on first read so UI is immediately usable
        default_kb = KnowledgeBaseModel(
            id="default",
            name="Campus & AI Open Lab Knowledge Base",
            description="Built-in bilingual corpus covering graduate regulations, GPU cluster Slurm guide, Medical AI compliance, and VeloRAG architecture.",
            chunk_size=300,
            chunk_overlap=40,
        )
        db.add(default_kb)
        db.commit()
        db.refresh(default_kb)
        return [default_kb]
    return kbs


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return kb


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_kb(kb_id: str, kb_update: KnowledgeBaseUpdate, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    update_data = kb_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(kb, key, value)

    db.commit()
    db.refresh(kb)
    return kb


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    db.delete(kb)
    db.commit()
    return {"status": "deleted"}


@router.post("/{kb_id}/documents", response_model=DocumentResponse)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb_dir = UPLOAD_DIR / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)
    file_path = kb_dir / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = os.path.getsize(file_path)

    doc = DocumentModel(
        kb_id=kb_id,
        filename=file.filename,
        file_path=str(file_path),
        file_size=file_size,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Zero-dependency synchronous fallback when Redis/Celery is not running
    use_celery = os.getenv("VELORAG_USE_CELERY", "0") == "1"
    if use_celery:
        try:
            from velorag.tasks.document_tasks import process_document
            process_document.delay(doc.id)
            return doc
        except Exception:
            pass

    try:
        doc.status = DocumentStatus.PARSING
        db.commit()
        rag = get_default_pipeline()
        chunk_count = rag.ingest(str(file_path))
        doc.chunk_count = chunk_count
        doc.status = DocumentStatus.INDEXED
        db.commit()
        db.refresh(doc)
    except Exception as e:
        doc.status = DocumentStatus.FAILED
        doc.error_msg = str(e)
        db.commit()
        db.refresh(doc)

    return doc


@router.post("/{kb_id}/texts", response_model=DocumentResponse)
def ingest_raw_text_document(
    kb_id: str,
    payload: DirectTextCreate,
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    kb_dir = UPLOAD_DIR / kb_id
    kb_dir.mkdir(parents=True, exist_ok=True)
    file_path = kb_dir / payload.filename
    file_path.write_text(payload.content, encoding="utf-8")

    rag = get_default_pipeline()
    chunk_count = rag.ingest_text(
        text=payload.content,
        source=payload.filename,
        metadata={"kb_id": kb_id, "title": payload.filename},
    )
    doc = DocumentModel(
        kb_id=kb_id,
        filename=payload.filename,
        file_path=str(file_path),
        file_size=len(payload.content.encode("utf-8")),
        chunk_count=chunk_count,
        status=DocumentStatus.INDEXED,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
def list_documents(kb_id: str, db: Session = Depends(get_db)):
    docs = db.query(DocumentModel).filter(DocumentModel.kb_id == kb_id).all()
    if not docs and kb_id == "default":
        from velorag.eval.dataset import get_builtin_campus_benchmark
        builtin_docs, _ = get_builtin_campus_benchmark()
        seeded = []
        for bdoc in builtin_docs:
            dm = DocumentModel(
                id=bdoc.doc_id,
                kb_id="default",
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
