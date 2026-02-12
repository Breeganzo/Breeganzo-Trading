from __future__ import annotations

import base64
import io
import logging
import urllib.parse
from datetime import datetime, timedelta, timezone

import qrcode
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from jose import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.database import get_db
from app.middleware.auth import get_current_user
from app.models.models import User
from app.schemas.schemas import TOTPVerify, TokenResponse, UserResponse
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── GET /login ────────────────────────────────────────────────────────
@router.get("/login")
async def login():
    """Return the Google OAuth 2.0 authorization URL.

    The frontend should redirect the user to this URL to begin the
    login flow.
    """
    auth_url = auth_service.get_google_auth_url()
    return {"auth_url": auth_url}


# ── GET /callback ─────────────────────────────────────────────────────
@router.get("/callback")
async def google_callback(
    code: str,
    db: AsyncSession = Depends(get_db),
):
    """Handle the Google OAuth 2.0 callback.

    Exchanges the authorization *code* for tokens, validates the user
    against the allowed email, creates or updates the user record,
    and redirects the browser to the frontend with the JWT token
    encoded in the URL fragment.
    """
    settings = get_settings()
    frontend_url = settings.ALLOWED_ORIGINS.split(",")[0].strip()  # e.g. http://localhost:3000

    try:
        result = await auth_service.exchange_google_code(code, db)
    except PermissionError as exc:
        logger.warning("Forbidden login: %s", exc)
        return RedirectResponse(
            url=f"{frontend_url}/login?error={urllib.parse.quote(str(exc))}"
        )
    except Exception as exc:
        logger.exception("Google OAuth callback failed")
        return RedirectResponse(
            url=f"{frontend_url}/login?error=google_auth_failed"
        )

    # Build redirect URL with token data as query params
    params = {
        "token": result["access_token"],
        "requires_totp": str(result.get("requires_totp", False)).lower(),
    }
    if result.get("totp_setup_uri"):
        params["totp_setup_uri"] = result["totp_setup_uri"]

    redirect_url = f"{frontend_url}/login?{urllib.parse.urlencode(params)}"
    return RedirectResponse(url=redirect_url)


# ── GET /callback/api ─────────────────────────────────────────────────
@router.get("/callback/api", response_model=TokenResponse)
async def google_callback_api(
    code: str,
    db: AsyncSession = Depends(get_db),
):
    """API-only callback (returns JSON instead of redirecting)."""
    try:
        result = await auth_service.exchange_google_code(code, db)
    except PermissionError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Google OAuth callback failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to authenticate with Google. Please try again.",
        )

    return TokenResponse(
        access_token=result["access_token"],
        token_type=result["token_type"],
        requires_totp=result.get("requires_totp", False),
        totp_setup_uri=result.get("totp_setup_uri"),
    )


# ── POST /totp/setup ─────────────────────────────────────────────────
@router.post("/totp/setup")
async def totp_setup(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a new TOTP secret for the current user.

    Returns the secret, the ``otpauth://`` provisioning URI, and a
    base64-encoded PNG QR code that can be scanned by an authenticator
    app such as Google Authenticator.

    This endpoint may be called again to reset TOTP before it is
    verified/enabled.
    """
    try:
        totp_data = await auth_service.setup_totp(current_user.id, db)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    # Generate QR code as base64-encoded PNG
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(totp_data["uri"])
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "secret": totp_data["secret"],
        "uri": totp_data["uri"],
        "qr_code": qr_base64,
    }


# ── POST /totp/verify ────────────────────────────────────────────────
@router.post("/totp/verify")
async def totp_verify(
    body: TOTPVerify,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Verify a six-digit TOTP code for the current user.

    On first verification this also enables TOTP for the account.
    A new JWT is returned that includes a ``totp_verified`` claim,
    granting the client full access to TOTP-protected endpoints.
    """
    if not current_user.totp_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="TOTP has not been set up. Call /auth/totp/setup first.",
        )

    # First-time enable: verify and flip the totp_enabled flag
    if not current_user.totp_enabled:
        success = await auth_service.verify_and_enable_totp(
            current_user.id, body.code, db
        )
        if not success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP code",
            )
    else:
        # Subsequent logins: just validate the code
        if not auth_service.verify_totp(current_user.totp_secret, body.code):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid TOTP code",
            )

    # Issue a new JWT that includes the totp_verified claim
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "user_id": str(current_user.id),
        "email": current_user.email,
        "exp": now + timedelta(minutes=settings.JWT_EXPIRY_MINUTES),
        "iat": now,
        "totp_verified": True,
    }
    access_token = jwt.encode(
        payload,
        settings.SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "totp_verified": True,
    }


# ── GET /me ───────────────────────────────────────────────────────────
@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        picture=current_user.picture,
        totp_enabled=current_user.totp_enabled,
        is_active=current_user.is_active,
    )
