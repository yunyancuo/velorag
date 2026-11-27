from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from velorag.db.models import User
from velorag.knowledge.models import KnowledgeBaseModel

DEFAULT_KB_ID = "default"


def ensure_default_kb(db: Session) -> KnowledgeBaseModel:
    """幂等确保内置公开演示知识库存在。"""
    kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == DEFAULT_KB_ID).first()
    if kb is not None:
        return kb
    kb = KnowledgeBaseModel(
        id=DEFAULT_KB_ID,
        name="Campus & AI Open Lab Knowledge Base",
        description=(
            "Built-in bilingual corpus covering graduate regulations, GPU cluster "
            "Slurm guide, Medical AI compliance, and VeloRAG architecture."
        ),
        owner_id=None,
        is_public=True,
        chunk_strategy="fixed",
        chunk_size=300,
        chunk_overlap=40,
        retrieval_mode="graph_enhanced",
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


def can_read_kb(kb: KnowledgeBaseModel, user: Optional[User]) -> bool:
    """判断用户是否具备知识库读取权限。"""
    if kb.is_public:
        return True
    if user is None:
        return False
    return bool(user.is_superuser or (kb.owner_id is not None and kb.owner_id == user.id))


def can_write_kb(kb: KnowledgeBaseModel, user: Optional[User]) -> bool:
    """判断用户是否具备知识库写入/修改/删除权限。"""
    if kb.owner_id is None:
        return True
    if user is None:
        return False
    return bool(user.is_superuser or kb.owner_id == user.id)


def authorize_kb_access(
    db: Session,
    kb_id: str,
    user: Optional[User],
    *,
    write: bool = False,
    allow_unregistered: bool = False,
) -> Optional[KnowledgeBaseModel]:
    """统一校验用户对指定知识库的读/写访问权限。"""
    if kb_id == DEFAULT_KB_ID:
        kb = ensure_default_kb(db)
    else:
        kb = db.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.id == kb_id).first()

    if kb is None:
        if allow_unregistered:
            return None
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    allowed = can_write_kb(kb, user) if write else can_read_kb(kb, user)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this knowledge base",
        )
    return kb


def get_restricted_private_kb_ids(db: Session, user: Optional[User]) -> set[str]:
    """返回当前用户无权访问的全部私有知识库 ID 集合，用于未指定 kb_id 时的防越权隔离。"""
    if user is not None and user.is_superuser:
        return set()

    q = db.query(KnowledgeBaseModel.id, KnowledgeBaseModel.owner_id).filter(
        KnowledgeBaseModel.is_public.is_(False)
    )
    restricted: set[str] = set()
    for kb_id, owner_id in q.all():
        if user is None or owner_id != user.id:
            restricted.add(kb_id)
    return restricted


def build_scoped_filters(
    db: Session,
    user: Optional[User],
    kb_id: Optional[str] = None,
    extra_filters: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """构造带多租户权限校验的检索过滤字典。"""
    filters: dict[str, Any] = dict(extra_filters or {})

    if kb_id:
        authorize_kb_access(db, kb_id, user, write=False, allow_unregistered=True)
        filters["kb_id"] = kb_id
        return filters

    # 未显式指定 kb_id 时，自动排除所有他人私有知识库，防范跨租户越权检索
    restricted = get_restricted_private_kb_ids(db, user)
    if restricted:
        filters["kb_id"] = {"$nin": sorted(restricted)}
    return filters
