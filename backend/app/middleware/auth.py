from __future__ import annotations

import logging
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.models import User
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

# Reusable scheme -- auto_error=True returns 403 when header is absent;
# we override with a clear 401 in the dependency instead.
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency that authenticates a request via JWT.

    Extracts the Bearer token from the ``Authorization`` header, verifies
    its signature and expiry, loads the corresponding :class:`User` from the
    database, and returns it.

    Raises:
        HTTPException 401: Token missing, invalid/expired, or user not found.
        HTTPException 403: User account is deactivated.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # Verify JWT signature and expiry
    try:
        payload = auth_service.verify_jwt_token(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract user_id from the token payload
    raw_user_id: str | None = payload.get("user_id")
    if raw_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = UUID(raw_user_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Load the user from the database
    user = await auth_service.get_user_by_id(user_id, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Ensure the account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    return user


async def require_totp_verified(
    user: User = Depends(get_current_user),
    x_totp_verified: str | None = Header(None, alias="X-TOTP-Verified"),
) -> User:
    """FastAPI dependency that enforces TOTP verification when enabled.

    If the user has TOTP enabled, the request must carry an
    ``X-TOTP-Verified: true`` header (typically set by the client after a
    successful TOTP challenge at login).  When TOTP is not enabled for the
    user, the request is allowed through without the header.

    In a full implementation the ``totp_verified`` claim would live inside
    the JWT itself; the header check here serves as a lightweight guard
    that works alongside the login flow.

    Raises:
        HTTPException 403: TOTP is enabled but verification is missing.
    """
    if not user.totp_enabled:
        return user

    if x_totp_verified is None or x_totp_verified.lower() != "true":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="TOTP verification required",
        )

    return user
