from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from velorag.auth.jwt import hash_password, verify_password, create_access_token, generate_api_key
from velorag.auth.dependencies import require_user
from velorag.db.deps import get_db
from velorag.db.models import User, ApiKey

router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=128)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class ApiKeyCreate(BaseModel):
    name: str = "default"


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    raw_key: str
    is_active: bool
    created_at: str


@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, session: Session = Depends(get_db)):
    existing = session.query(User).filter(
        (User.username == req.username) | (User.email == req.email)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username or email already exists")

    user = User(
        username=req.username,
        email=req.email,
        hashed_password=hash_password(req.password),
    )
    session.add(user)
    session.commit()
    session.refresh(user)

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, session: Session = Depends(get_db)):
    user = session.query(User).filter(User.username == req.username).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token(user.id)
    return TokenResponse(access_token=token, user_id=user.id, username=user.username)


@router.post("/api-keys", response_model=ApiKeyResponse)
def create_api_key(req: ApiKeyCreate, user: User = Depends(require_user), session: Session = Depends(get_db)):
    raw, key_hash, prefix = generate_api_key()

    api_key = ApiKey(
        user_id=user.id,
        name=req.name,
        key_hash=key_hash,
        key_prefix=prefix,
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
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
def list_api_keys(user: User = Depends(require_user), session: Session = Depends(get_db)):
    keys = session.query(ApiKey).filter(ApiKey.user_id == user.id).all()
    return [
        ApiKeyResponse(
            id=k.id, name=k.name, key_prefix=k.key_prefix,
            raw_key="", is_active=k.is_active, created_at=k.created_at.isoformat(),
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
def delete_api_key(key_id: str, user: User = Depends(require_user), session: Session = Depends(get_db)):
    key = session.query(ApiKey).filter(ApiKey.id == key_id, ApiKey.user_id == user.id).first()
    if not key:
        raise HTTPException(404, "API key not found")
    session.delete(key)
    session.commit()
    return {"status": "deleted"}


@router.get("/me", response_model=TokenResponse)
def me(user: User = Depends(require_user)):
    return TokenResponse(
        access_token="", user_id=user.id, username=user.username,
    )
