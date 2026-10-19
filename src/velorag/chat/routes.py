from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from velorag.chat.models import ChatMessageModel, ChatSessionModel
from velorag.chat.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatSessionCreate,
    ChatSessionResponse,
)
from velorag.core.database import get_db
from velorag.server.state import get_default_pipeline

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/sessions", response_model=ChatSessionResponse)
def create_session(session_in: ChatSessionCreate, db: Session = Depends(get_db)):
    db_session = ChatSessionModel(**session_in.model_dump())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(kb_id: str | None = None, db: Session = Depends(get_db)):
    query = db.query(ChatSessionModel)
    if kb_id:
        query = query.filter(ChatSessionModel.kb_id == kb_id)
    return query.order_by(ChatSessionModel.updated_at.desc()).all()


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)):
    session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(session)
    db.commit()
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def send_message(
    session_id: str,
    chat_req: ChatRequest,
    db: Session = Depends(get_db),
):
    session = db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_msg = ChatMessageModel(
        session_id=session_id,
        role="user",
        content=chat_req.message,
    )
    db.add(user_msg)
    db.commit()

    rag = get_default_pipeline()
    response = rag.query(
        query=chat_req.message,
        top_k=chat_req.top_k,
        mode=chat_req.mode,
        use_graph=(chat_req.mode in ("hybrid", "graph_enhanced")),
        system_prompt=session.system_prompt,
    )

    sources_data = [
        {
            "index": c.index,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source": c.source,
            "text": c.quote,
            "score": c.score,
        }
        for c in response.citations
    ]
    if not sources_data:
        sources_data = [
            {
                "chunk_id": s.chunk.id,
                "doc_id": s.chunk.doc_id,
                "source": s.chunk.metadata.get("source", s.retrieval_source),
                "text": s.chunk.text[:200],
                "score": round(s.score, 4),
            }
            for s in response.sources
        ]

    assistant_msg = ChatMessageModel(
        session_id=session_id,
        role="assistant",
        content=response.content,
        sources=sources_data,
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return assistant_msg
