from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional

from velorag.auth.dependencies import require_user
from velorag.db.deps import get_db
from velorag.db.models import User, KnowledgeBase, Document
from velorag.knowledge.service import KnowledgeBaseService
from velorag.tasks.ingestion import process_document

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge_bases"])


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    description: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_mode: str = "hybrid"
    graph_enabled: bool = False
    is_public: bool = False


class KBUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_strategy: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    retrieval_mode: Optional[str] = None
    graph_enabled: Optional[bool] = None
    is_public: Optional[bool] = None


class KBResponse(BaseModel):
    id: str
    name: str
    description: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    retrieval_mode: str
    graph_enabled: bool
    document_count: int = 0
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    created_at: str

    model_config = {"from_attributes": True}


@router.post("", response_model=KBResponse)
def create_kb(req: KBCreate, user: User = Depends(require_user), session: Session = Depends(get_db)):
    kb = KnowledgeBaseService.create(
        session, user,
        name=req.name,
        description=req.description,
        embedding_model=req.embedding_model,
        chunk_strategy=req.chunk_strategy,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        retrieval_mode=req.retrieval_mode,
        graph_enabled=req.graph_enabled,
        is_public=req.is_public,
    )
    return _kb_to_response(kb)


@router.get("", response_model=list[KBResponse])
def list_kbs(user: User = Depends(require_user), session: Session = Depends(get_db)):
    kbs = KnowledgeBaseService.list_for_user(session, user)
    return [_kb_to_response(kb) for kb in kbs]


@router.get("/{kb_id}", response_model=KBResponse)
def get_kb(kb_id: str, user: User = Depends(require_user), session: Session = Depends(get_db)):
    kb = KnowledgeBaseService.get(session, kb_id, user)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return _kb_to_response(kb)


@router.patch("/{kb_id}", response_model=KBResponse)
def update_kb(kb_id: str, req: KBUpdate, user: User = Depends(require_user), session: Session = Depends(get_db)):
    kb = KnowledgeBaseService.update(
        session, kb_id, user,
        **req.model_dump(exclude_none=True),
    )
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    return _kb_to_response(kb)


@router.delete("/{kb_id}")
def delete_kb(kb_id: str, user: User = Depends(require_user), session: Session = Depends(get_db)):
    ok = KnowledgeBaseService.delete(session, kb_id, user)
    if not ok:
        raise HTTPException(404, "Knowledge base not found")
    return {"status": "deleted"}


@router.post("/{kb_id}/documents", response_model=DocumentResponse)
async def upload_document(
    kb_id: str,
    file: UploadFile = File(...),
    user: User = Depends(require_user),
    session: Session = Depends(get_db),
):
    kb = KnowledgeBaseService.get(session, kb_id, user)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")

    import tempfile, shutil
    from pathlib import Path

    suffix = Path(file.filename).suffix.lower()
    allowed = {".txt", ".md", ".pdf", ".docx", ".markdown"}
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    doc = Document(
        kb_id=kb_id,
        filename=file.filename,
        file_type=suffix,
        file_size=Path(tmp_path).stat().st_size,
        status="pending",
    )
    session.add(doc)
    session.commit()
    session.refresh(doc)

    process_document.delay(doc.id, kb_id, tmp_path)

    return _doc_to_response(doc)


@router.get("/{kb_id}/documents", response_model=list[DocumentResponse])
def list_documents(kb_id: str, user: User = Depends(require_user), session: Session = Depends(get_db)):
    kb = KnowledgeBaseService.get(session, kb_id, user)
    if not kb:
        raise HTTPException(404, "Knowledge base not found")
    docs = KnowledgeBaseService.get_documents(session, kb_id)
    return [_doc_to_response(d) for d in docs]


def _kb_to_response(kb) -> KBResponse:
    doc_count = len(kb.documents) if kb.documents else 0
    return KBResponse(
        id=kb.id, name=kb.name, description=kb.description or "",
        embedding_model=kb.embedding_model, chunk_strategy=kb.chunk_strategy,
        chunk_size=kb.chunk_size, retrieval_mode=kb.retrieval_mode,
        graph_enabled=kb.graph_enabled, document_count=doc_count,
        created_at=kb.created_at.isoformat() if kb.created_at else "",
        updated_at=kb.updated_at.isoformat() if kb.updated_at else "",
    )


def _doc_to_response(doc) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id, filename=doc.filename, file_type=doc.file_type,
        file_size=doc.file_size, status=doc.status, chunk_count=doc.chunk_count,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
    )
