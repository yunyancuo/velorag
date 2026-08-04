from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from velorag.auth.jwt import decode_access_token
from velorag.db.base import get_session
from velorag.db.models import User, ApiKey

_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    session: Session = Depends(get_session),
) -> Optional[User]:
    if credentials is None:
        return None

    token = credentials.credentials

    # Try JWT first
    user_id = decode_access_token(token)
    if user_id:
        user = session.query(User).filter(User.id == user_id).first()
        if user and user.is_active:
            return user

    # Try API Key
    import hashlib
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    api_key = session.query(ApiKey).filter(
        ApiKey.key_hash == key_hash,
        ApiKey.is_active == True,
    ).first()

    if api_key:
        if api_key.expires_at and api_key.expires_at < __import__("datetime").datetime.now(__import__("datetime").timezone.utc):
            return None
        return session.query(User).filter(User.id == api_key.user_id).first()

    return None


def require_user(user: Optional[User] = Depends(get_current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return user


def require_superuser(user: User = Depends(require_user)) -> User:
    if not user.is_superuser:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Superuser required")
    return user
