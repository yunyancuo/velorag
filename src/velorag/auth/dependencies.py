from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from velorag.auth.jwt import decode_access_token, hash_api_key
from velorag.db.deps import get_db
from velorag.db.models import ApiKey, User

_bearer = HTTPBearer(auto_error=False)


def _ensure_utc(dt: datetime) -> datetime:
    """将数据库返回的 datetime 统一规范为 UTC 时区感知对象（兼容 SQLite）。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    session: Session = Depends(get_db),
) -> Optional[User]:
    """解析 Bearer JWT 或 API Key 并返回当前处于激活状态的用户。"""
    if credentials is None or not credentials.credentials:
        return None

    token = credentials.credentials.strip()
    if not token:
        return None

    # 优先尝试 JWT 访问令牌
    user_id = decode_access_token(token)
    if user_id:
        user = session.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            return user
        return None

    # 回退尝试 API Key 认证
    key_hash = hash_api_key(token)
    api_key = (
        session.query(ApiKey)
        .filter(
            ApiKey.key_hash == key_hash,
            ApiKey.is_active.is_(True),
        )
        .first()
    )

    if api_key is None:
        return None

    now = datetime.now(timezone.utc)
    if api_key.expires_at and _ensure_utc(api_key.expires_at) <= now:
        return None

    user = session.query(User).filter(User.id == api_key.user_id).first()
    if user is None or not user.is_active:
        return None

    api_key.last_used_at = now
    session.commit()
    return user


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_superuser(user: User = Depends(require_user)) -> User:
    if not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superuser required",
        )
    return user
