from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from velorag.auth.dependencies import get_current_user
from velorag.chat.models import ChatMessageModel, ChatSessionModel
from velorag.chat.schemas import (
    ChatMessageResponse,
    ChatRequest,
    ChatSessionCreate,
    ChatSessionResponse,
)
from velorag.core.database import get_db
from velorag.db.models import User
from velorag.knowledge.access import authorize_kb_access
from velorag.server.state import get_default_pipeline

router = APIRouter(prefix="/chat", tags=["Chat"])


def _authorize_session(
    session: Optional[ChatSessionModel], user: Optional[User]
) -> ChatSessionModel:
    """校验聊天会话存在且当前用户有权访问。"""
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
        )
    if session.owner_id is not None:
        if user is None or (not user.is_superuser and session.owner_id != user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this chat session",
            )
    return session


@router.post("/sessions", response_model=ChatSessionResponse)
def create_session(
    session_in: ChatSessionCreate,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    authorize_kb_access(db, session_in.kb_id, user, write=False)
    db_session = ChatSessionModel(
        **session_in.model_dump(),
        owner_id=user.id if user is not None else None,
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


@router.get("/sessions", response_model=list[ChatSessionResponse])
def list_sessions(
    kb_id: str | None = None,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if kb_id:
        authorize_kb_access(db, kb_id, user, write=False)

    query = db.query(ChatSessionModel)
    if kb_id:
        query = query.filter(ChatSessionModel.kb_id == kb_id)

    if user is None:
        query = query.filter(ChatSessionModel.owner_id.is_(None))
    elif not user.is_superuser:
        query = query.filter(
            ChatSessionModel.owner_id.is_(None)
            | (ChatSessionModel.owner_id == user.id)
        )
    return query.order_by(ChatSessionModel.updated_at.desc()).all()


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
def get_session(
    session_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    )
    session = _authorize_session(session, user)
    authorize_kb_access(db, session.kb_id, user, write=False)
    return session


@router.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    )
    session = _authorize_session(session, user)
    db.delete(session)
    db.commit()
    return {"status": "deleted"}


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def send_message(
    session_id: str,
    chat_req: ChatRequest,
    user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = (
        db.query(ChatSessionModel).filter(ChatSessionModel.id == session_id).first()
    )
    session = _authorize_session(session, user)
    # 每次发送消息前再次校验所绑定知识库的读取权限，防止权限回收后继续越权读取
    authorize_kb_access(db, session.kb_id, user, write=False)

    user_msg = ChatMessageModel(
        session_id=session_id,
        role="user",
        content=chat_req.message,
    )
    db.add(user_msg)
    db.commit()

    rag = get_default_pipeline()
    # 严格按 session.kb_id 作用域过滤，阻断跨知识库数据泄漏
    response = rag.query(
        query=chat_req.message,
        top_k=chat_req.top_k,
        mode=chat_req.mode,
        use_graph=(chat_req.mode == "graph_enhanced"),
        system_prompt=session.system_prompt,
        filters={"kb_id": session.kb_id},
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
