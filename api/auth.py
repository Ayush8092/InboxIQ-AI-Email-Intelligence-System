"""
Production JWT authentication.
- Short-lived access tokens (15 minutes)
- Long-lived refresh tokens (7 days)
- Token blacklist for revocation
- Secure middleware design
- Token type validation
"""
import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from utils.secure_logger import get_secure_logger

logger   = get_secure_logger(__name__)
security = HTTPBearer(auto_error=False)

JWT_SECRET         = os.getenv("JWT_SECRET", os.urandom(32).hex())
JWT_ALGORITHM      = "HS256"
ACCESS_TOKEN_EXP   = 15          # minutes
REFRESH_TOKEN_EXP  = 7 * 24 * 60 # minutes (7 days)

# In-memory token blacklist (use Redis in multi-instance production)
_blacklist: set[str] = set()

ROLES = {
    "admin":  ["read","write","train","admin","process_all"],
    "user":   ["read","write"],
    "viewer": ["read"],
}


def _hash_token(token: str) -> str:
    """Hash token for blacklist storage (don't store raw tokens)."""
    return hashlib.sha256(token.encode()).hexdigest()


def create_access_token(user_id: str, role: str = "user") -> str:
    """Create short-lived access token (15 min)."""
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed")
    payload = {
        "sub":   user_id,
        "role":  role,
        "type":  "access",
        "exp":   datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXP),
        "iat":   datetime.utcnow(),
        "jti":   os.urandom(8).hex(),  # unique token ID
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str, role: str = "user") -> str:
    """Create long-lived refresh token (7 days)."""
    try:
        from jose import jwt
    except ImportError:
        raise RuntimeError("python-jose not installed")
    payload = {
        "sub":  user_id,
        "role": role,
        "type": "refresh",
        "exp":  datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXP),
        "iat":  datetime.utcnow(),
        "jti":  os.urandom(8).hex(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    """
    Decode and validate JWT token.
    Checks: signature, expiry, blacklist.
    """
    try:
        from jose import jwt, JWTError, ExpiredSignatureError
    except ImportError:
        return None

    # Check blacklist first
    if _hash_token(token) in _blacklist:
        logger.warning("Blacklisted token used")
        return None

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except ExpiredSignatureError:
        logger.debug("Expired token rejected")
        return None
    except JWTError as e:
        logger.warning(f"Invalid token: {type(e).__name__}")
        return None


def revoke_token(token: str):
    """Add token to blacklist (immediate revocation)."""
    _blacklist.add(_hash_token(token))
    logger.info("Token revoked")


def require_permission(permission: str):
    """
    FastAPI dependency for RBAC permission checking.
    Validates JWT, token type (access only), and role permissions.
    """
    def checker(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
        x_api_key: Optional[str] = Header(None),
    ) -> dict:
        # Service-to-service API key
        api_key_env = os.getenv("AEOA_API_KEY", "")
        if api_key_env and x_api_key == api_key_env:
            return {"sub": "service", "role": "admin"}

        if not credentials:
            raise HTTPException(
                status_code=401,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = decode_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Reject refresh tokens used as access tokens
        if payload.get("type") == "refresh":
            raise HTTPException(
                status_code=401,
                detail="Refresh token cannot be used for API access",
            )

        role  = payload.get("role", "viewer")
        perms = ROLES.get(role, [])

        if permission not in perms:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' lacks permission '{permission}'",
            )

        return payload

    return checker


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None),
) -> dict:
    """Get current user from JWT — minimal auth check."""
    api_key_env = os.getenv("AEOA_API_KEY", "")
    if api_key_env and x_api_key == api_key_env:
        return {"sub": "service", "role": "admin"}

    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if payload.get("type") == "refresh":
        raise HTTPException(status_code=401, detail="Use access token")

    return payload