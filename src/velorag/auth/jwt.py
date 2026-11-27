from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from velorag.core.settings import ALLOWED_JWT_ALGORITHMS, get_settings

# 为支持超过 bcrypt 72 字节限制的长密码并兼容标准 bcrypt 哈希，采用显式前缀区分预哈希方案
_PREHASH_PREFIX = "velorag-sha256-bcrypt$"


def _validate_algorithm(algorithm: str) -> str:
    """校验 JWT 签名算法位于安全白名单内，防止 alg=none 及算法混淆攻击。"""
    if algorithm not in ALLOWED_JWT_ALGORITHMS:
        raise ValueError(f"不允许的 JWT 算法: {algorithm!r}")
    return algorithm


def hash_password(password: str) -> str:
    """使用 bcrypt 对密码进行加盐哈希，对超长密码自动进行 SHA-256 预哈希避免截断风险。"""
    if not isinstance(password, str) or not password:
        raise ValueError("密码不能为空")
    raw = password.encode("utf-8")
    if len(raw) > 72:
        digest = hashlib.sha256(raw).hexdigest().encode("ascii")
        hashed = bcrypt.hashpw(digest, bcrypt.gensalt(rounds=12)).decode("ascii")
        return f"{_PREHASH_PREFIX}{hashed}"
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """安全校验明文密码与哈希值，捕获异常格式以防产生未处理异常。"""
    if not plain or not hashed:
        return False
    try:
        raw = plain.encode("utf-8")
        if hashed.startswith(_PREHASH_PREFIX):
            bcrypt_hash = hashed.removeprefix(_PREHASH_PREFIX).encode("ascii")
            digest = hashlib.sha256(raw).hexdigest().encode("ascii")
            return bcrypt.checkpw(digest, bcrypt_hash)
        if len(raw) > 72:
            return False
        return bcrypt.checkpw(raw, hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """签发带签发时间、过期时间与唯一标识（jti）的访问 JWT。"""
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id 必须为非空字符串")
    settings = get_settings()
    algorithm = _validate_algorithm(settings.auth.algorithm)
    now = datetime.now(timezone.utc)
    expire = now + (
        expires_delta or timedelta(minutes=settings.auth.access_token_expire_minutes)
    )
    payload = {
        "sub": user_id,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "jti": uuid.uuid4().hex,
        "type": "access",
    }
    return jwt.encode(payload, settings.auth.secret_key, algorithm=algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """校验访问 JWT 的签名、算法白名单、过期时间与令牌类型，返回用户 ID。"""
    if not token or not isinstance(token, str):
        return None
    settings = get_settings()
    try:
        algorithm = _validate_algorithm(settings.auth.algorithm)
        # 显式检查 JWT Header 防止 alg=none 或算法降级攻击
        header = jwt.get_unverified_header(token)
        if header.get("alg") != algorithm:
            return None
        payload = jwt.decode(
            token,
            settings.auth.secret_key,
            algorithms=[algorithm],
            options={"require_exp": True, "require_sub": True},
        )
        if payload.get("type") != "access":
            return None
        sub = payload.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            return None
        return sub
    except (JWTError, ValueError, TypeError):
        return None


def hash_api_key(raw_key: str) -> str:
    """计算 API Key 的 SHA-256 摘要用于持久化存储与比对。"""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def verify_api_key_hash(raw_key: str, expected_hash: str) -> bool:
    """使用常量时间比较验证 API Key 哈希，抵御时序侧信道攻击。"""
    return hmac.compare_digest(hash_api_key(raw_key), expected_hash)


def generate_api_key() -> tuple[str, str, str]:
    """生成带平台前缀的高熵 API Key，返回 (raw_key, key_hash, key_prefix)。"""
    settings = get_settings()
    token_body = secrets.token_hex(32)
    raw = f"{settings.auth.api_key_prefix}{token_body}"
    key_hash = hash_api_key(raw)
    prefix = raw[: min(len(raw), 12)]
    return raw, key_hash, prefix
