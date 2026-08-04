from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from velorag.db.models import KnowledgeBase, Document, User


class KnowledgeBaseService:

    @staticmethod
    def create(session: Session, owner: User, name: str, description: str = "", **kwargs) -> KnowledgeBase:
        kb = KnowledgeBase(
            name=name,
            description=description,
            owner_id=owner.id,
            **kwargs,
        )
        session.add(kb)
        session.commit()
        session.refresh(kb)
        return kb

    @staticmethod
    def get(session: Session, kb_id: str, owner: Optional[User] = None) -> Optional[KnowledgeBase]:
        q = session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id)
        if owner:
            q = q.filter(
                (KnowledgeBase.owner_id == owner.id) | (KnowledgeBase.is_public == True)
            )
        kb = q.first()
        return kb

    @staticmethod
    def list_for_user(session: Session, owner: User) -> list[KnowledgeBase]:
        return session.query(KnowledgeBase).filter(
            KnowledgeBase.owner_id == owner.id
        ).order_by(KnowledgeBase.updated_at.desc()).all()

    @staticmethod
    def update(session: Session, kb_id: str, owner: User, **kwargs) -> Optional[KnowledgeBase]:
        kb = KnowledgeBaseService.get(session, kb_id, owner)
        if not kb:
            return None
        for key, value in kwargs.items():
            if hasattr(kb, key) and value is not None:
                setattr(kb, key, value)
        session.commit()
        session.refresh(kb)
        return kb

    @staticmethod
    def delete(session: Session, kb_id: str, owner: User) -> bool:
        kb = session.query(KnowledgeBase).filter(
            KnowledgeBase.id == kb_id, KnowledgeBase.owner_id == owner.id
        ).first()
        if not kb:
            return False
        session.delete(kb)
        session.commit()
        return True

    @staticmethod
    def get_documents(session: Session, kb_id: str) -> list[Document]:
        return session.query(Document).filter(
            Document.kb_id == kb_id
        ).order_by(Document.created_at.desc()).all()
