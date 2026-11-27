from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from velorag.auth.dependencies import require_user
from velorag.auth.jwt import (
    create_access_token,
    generate_api_key,
    hash_password,
    verify_password,
)
from velorag.db.deps import get_db
from velorag.db.models import ApiKey, User

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=128)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class ApiKeyCreate(BaseModel):
    name: str = Field(default="default", min_length=1, max_length=128)
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=3650)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    raw_key: str
    is_active: bool
    created_at: str
    expires_at: Optional[str] = None


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, session: Session = Depends(get_db)):
    username = req.username.strip()
    email = str(req.email).strip().lower()
    if len(username) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Username must contain at least 3 non-whitespace characters",
        )

    existing = (
        session.query(User)
        .filter((User.username == username) | (User.email == email))
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        )

    user = User(
        username=username,
        email=email,
        hashed_password=hash_password(req.password),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already exists",
        ) from None
    session.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, session: Session = Depends(get_db)):
    user = session.query(User).filter(User.username == req.username.strip()).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account disabled",
        )

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/api-keys", response_model=ApiKeyResponse)
def create_api_key(
    req: ApiKeyCreate,
    user: User = Depends(require_user),
    session: Session = Depends(get_db),
):
    raw, key_hash, prefix = generate_api_key()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(days=req.expires_in_days)
        if req.expires_in_days is not None
        else None
    )

    api_key = ApiKey(
        user_id=user.id,
        name=req.name.strip(),
        key_hash=key_hash,
        key_prefix=prefix,
        expires_at=expires_at,
    )
    session.add(api_key)
    session.commit()
    session.refresh(api_key)

    return ApiKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        raw_key=raw,
        is_active=api_key.is_active,
        created_at=api_key.created_at.isoformat(),
        expires_at=api_key.expires_at.isoformat() if api_key.expires_at else None,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
def list_api_keys(
    user: User = Depends(require_user),
    session: Session = Depends(get_db),
):
    keys = session.query(ApiKey).filter(ApiKey.user_id == user.id).all()
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            raw_key="",
            is_active=k.is_active,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
def delete_api_key(
    key_id: str,
    user: User = Depends(require_user),
    session: Session = Depends(get_db),
):
    key = (
        session.query(ApiKey)
        .filter(ApiKey.id == key_id, ApiKey.user_id == user.id)
        .first()
    )
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
    session.delete(key)
    session.commit()
    return {"status": "deleted"}


@router.get("/me", response_model=TokenResponse)
def me(user: User = Depends(require_user)):
    return TokenResponse(
        access_token="",
        user_id=user.id,
        username=user.username,
    )
