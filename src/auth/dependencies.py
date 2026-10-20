"""
FastAPI Authentication Dependencies for Past2Pixel.
Extracts and validates JWT Bearer tokens to enforce user-level access controls.
"""
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .security import decode_access_token
from ..db.mongodb import db_manager

# OAuth2 bearer token extractor
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Dict[str, Any]:
    """
    Validates JWT token and retrieves the authenticated user document.
    Raises 401 Unauthorized if token is missing, invalid, expired, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or authentication token expired.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    email: Optional[str] = payload.get("email")

    if not user_id and not email:
        raise credentials_exception

    user = None
    if user_id:
        user = db_manager.get_user_by_id(user_id)
    if not user and email:
        user = db_manager.get_user_by_email(email)

    if not user:
        raise credentials_exception

    # Remove password_hash before returning to handlers
    safe_user = user.copy()
    safe_user.pop("password_hash", None)
    return safe_user


async def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[Dict[str, Any]]:
    """Returns current user if valid token present, or None for anonymous requests."""
    if not token:
        return None
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
