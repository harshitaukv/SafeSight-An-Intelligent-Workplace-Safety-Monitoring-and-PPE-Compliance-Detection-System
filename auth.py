"""
Authentication for Safesight.

Two roles:
  - "admin": can upload documents, trigger knowledge-base rebuilds, and use
    the RAG chat to get results back from processed documents.
  - "user":  normal login, only gets the PPE-detection dashboard.

Uses bcrypt for password hashing and a signed JWT for session tokens.
Install deps once:  pip install pyjwt bcrypt
"""

import os
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from database import users

# ----------------------------------------------------
# Config
# ----------------------------------------------------
# In production set this via an environment variable instead of hardcoding it.
SECRET_KEY = os.environ.get("SAFESIGHT_SECRET_KEY", "safesight-dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

security = HTTPBearer()


# ----------------------------------------------------
# Password helpers
# ----------------------------------------------------
def _password_bytes(plain_password: str) -> bytes:
    """
    bcrypt only considers the first 72 bytes of a password, and newer
    releases raise ValueError instead of truncating silently. verify_password
    catches that and returns False — so a long but perfectly correct password
    came back as "invalid". Truncating identically on both sides keeps
    hashing and verification in agreement.
    """
    return (plain_password or "").encode("utf-8")[:72]


def hash_password(plain_password: str) -> str:
    return bcrypt.hashpw(_password_bytes(plain_password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(_password_bytes(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------
# JWT helpers
# ----------------------------------------------------
def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication token.")


# ----------------------------------------------------
# FastAPI dependencies
# ----------------------------------------------------
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Any logged-in user (admin or normal user)."""
    payload = decode_access_token(credentials.credentials)
    return {"username": payload["sub"], "role": payload["role"]}


def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """Admin-only routes (document upload, knowledge base rebuild)."""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user


# ----------------------------------------------------
# Default accounts (created once if the users collection is empty)
# ----------------------------------------------------
def seed_default_users():
    if users.count_documents({}) > 0:
        return

    default_accounts = [
        {"username": "admin", "password": "admin123", "role": "admin"},
        {"username": "user", "password": "user123", "role": "user"},
    ]

    for account in default_accounts:
        users.insert_one(
            {
                "username": account["username"],
                "password_hash": hash_password(account["password"]),
                "role": account["role"],
                "created_at": str(datetime.now()),
            }
        )

    print(
        "Seeded default accounts -> admin/admin123 (role: admin), "
        "user/user123 (role: user). Change these passwords in production."
    )
