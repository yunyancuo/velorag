from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from velorag.db.models import Document, KnowledgeBase, User

_UPDATABLE_KB_FIELDS = frozenset(
    {
        "name",
        "description",
        "embedding_model",
        "embedding_dim",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "retrieval_mode",
        "graph_enabled",
        "is_public",
    }
)


class KnowledgeBaseService:
    """多租户知识库服务层，严格区分读取权限与写操作所有权校验。"""

    @staticmethod
    def create(session: Session, owner: User, name: str, description: str = "", **kwargs) -> KnowledgeBase:
        safe_kwargs = {k: v for k, v in kwargs.items() if k in _UPDATABLE_KB_FIELDS}
        kb = KnowledgeBase(
            name=name,
            description=description,
            owner_id=owner.id,
            **safe_kwargs,
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)
        return kb

    @staticmethod
    def get(session: Session, kb_id: str, owner: Optional[User] = None) -> Optional[KnowledgeBase]:
        q = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id)
        if owner and not owner.is_superuser:
            q = q.filter(
                (KnowledgeBase.owner_id == owner.id) | KnowledgeBase.is_public.is_(True)
            )
        elif owner is None:
            q = q.filter(KnowledgeBase.is_public.is_(True))
        return q.first()

    @staticmethod
    def list_for_user(session: Session, owner: User) -> list[KnowledgeBase]:
        return (
            session.query(KnowledgeBase)
            .filter(KnowledgeBase.owner_id == owner.id)
            .order_by(KnowledgeBase.updated_at.desc())
            .all()
        )

    @staticmethod
    def update(session: Session, kb_id: str, owner: User, **kwargs) -> Optional[KnowledgeBase]:
        # 写操作必须严格校验所有权（或超级管理员），禁止修改他人公开知识库
        q = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id)
        if not owner.is_superuser:
            q = q.filter(KnowledgeBase.owner_id == owner.id)
        kb = q.first()
        if not kb:
            return None
        for key, value in kwargs.items():
            if key in _UPDATABLE_KB_FIELDS and value is not None:
                setattr(kb, key, value)
        session.commit()
        session.refresh(kb)
        return kb

    @staticmethod
    def delete(session: Session, kb_id: str, owner: User) -> bool:
        q = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id)
        if not owner.is_superuser:
            q = q.filter(KnowledgeBase.owner_id == owner.id)
        kb = q.first()
        if not kb:
            return False
        session.delete(kb)
        session.commit()
        return True

    @staticmethod
    def get_documents(
        session: Session, kb_id: str, owner: Optional[User] = None
    ) -> list[Document]:
        kb = KnowledgeBaseService.get(session, kb_id, owner=owner)
        if not kb:
            return []
        return (
            session.query(Document)
            .filter(Document.kb_id == kb_id)
            .order_by(Document.created_at.desc())
            .all()
        )
