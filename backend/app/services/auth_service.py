from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID

import httpx
import pyotp
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.models import User

logger = logging.getLogger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


class AuthService:
    """Authentication service for single-user QuantDesk Pro platform.

    Handles Google OAuth 2.0 login, TOTP-based two-factor authentication,
    and JWT session management. Only the email specified in ALLOWED_EMAIL
    is permitted to authenticate.
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Google OAuth 2.0 ──────────────────────────────────────────────

    def get_google_auth_url(self) -> str:
        """Build the Google OAuth 2.0 authorization URL."""
        params = {
            "client_id": self._settings.GOOGLE_CLIENT_ID,
            "redirect_uri": self._settings.GOOGLE_REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "consent",
        }
        url = str(httpx.URL(GOOGLE_AUTH_URL).copy_merge_params(params))
        return url

    async def exchange_google_code(self, code: str, db: AsyncSession) -> dict:
        """Exchange an authorization code for tokens, validate the user,
        and return user data together with a JWT access token.

        Raises ``PermissionError`` if the Google account email does not
        match ``ALLOWED_EMAIL``.
        """
        # 1. Exchange authorization code for Google tokens
        token_payload = {
            "code": code,
            "client_id": self._settings.GOOGLE_CLIENT_ID,
            "client_secret": self._settings.GOOGLE_CLIENT_SECRET,
            "redirect_uri": self._settings.GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            token_resp = await client.post(GOOGLE_TOKEN_URL, data=token_payload)
            token_resp.raise_for_status()
            token_data = token_resp.json()

            # 2. Fetch user information from Google
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            userinfo_resp = await client.get(GOOGLE_USERINFO_URL, headers=headers)
            userinfo_resp.raise_for_status()
            userinfo = userinfo_resp.json()

        email: str = userinfo.get("email", "").lower().strip()
        name: str | None = userinfo.get("name")
        picture: str | None = userinfo.get("picture")

        # 3. Enforce single-user restriction
        if email != self._settings.ALLOWED_EMAIL.lower().strip():
            logger.warning("Unauthorized login attempt from %s", email)
            raise PermissionError(
                f"Access denied. Only {self._settings.ALLOWED_EMAIL} may log in."
            )

        # 4. Fetch or create the user
        user = await self.get_user_by_email(email, db)
        if user is None:
            user = User(email=email, name=name, picture=picture)
            db.add(user)
            await db.flush()
            await db.refresh(user)
            logger.info("Created new user record for %s", email)
        else:
            # Update profile data from Google on each login
            user.name = name or user.name
            user.picture = picture or user.picture
            await db.flush()

        # 5. Build response
        access_token = self.create_jwt_token(str(user.id), user.email)

        result: dict = {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name,
                "picture": user.picture,
                "totp_enabled": user.totp_enabled,
                "is_active": user.is_active,
            },
            "access_token": access_token,
            "token_type": "bearer",
            "requires_totp": user.totp_enabled,
        }

        # If TOTP has never been set up, include a setup URI so the
        # frontend can present a QR code on first login.
        if not user.totp_enabled and not user.totp_secret:
            totp_data = await self.setup_totp(user.id, db)
            result["totp_setup_uri"] = totp_data["uri"]

        return result

    # ── JWT Session Management ────────────────────────────────────────

    def create_jwt_token(self, user_id: str, email: str) -> str:
        """Create a signed JWT containing *user_id*, *email*, and *exp*."""
        now = datetime.now(timezone.utc)
        payload = {
            "user_id": user_id,
            "email": email,
            "exp": now + timedelta(minutes=self._settings.JWT_EXPIRY_MINUTES),
            "iat": now,
        }
        return jwt.encode(
            payload,
            self._settings.SECRET_KEY,
            algorithm=self._settings.JWT_ALGORITHM,
        )

    def verify_jwt_token(self, token: str) -> dict:
        """Decode and verify a JWT.  Returns the payload dict.

        Raises ``JWTError`` on invalid or expired tokens.
        """
        try:
            payload: dict = jwt.decode(
                token,
                self._settings.SECRET_KEY,
                algorithms=[self._settings.JWT_ALGORITHM],
            )
            return payload
        except JWTError:
            logger.warning("JWT verification failed")
            raise

    # ── TOTP Two-Factor Authentication ────────────────────────────────

    @staticmethod
    def generate_totp_secret() -> str:
        """Generate a new random base-32 TOTP secret."""
        return pyotp.random_base32()

    def get_totp_uri(self, secret: str, email: str) -> str:
        """Return a ``otpauth://`` provisioning URI for Google Authenticator."""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name="QuantDesk Pro")

    @staticmethod
    def verify_totp(secret: str, code: str) -> bool:
        """Verify a six-digit TOTP code with a window of 1 to allow for
        slight clock drift."""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=1)

    async def setup_totp(self, user_id: UUID, db: AsyncSession) -> dict:
        """Generate a new TOTP secret for the user and persist it.

        Returns a dict with ``secret`` and ``uri`` (for QR rendering).
        TOTP is *not* enabled until ``verify_and_enable_totp`` succeeds.
        """
        user = await self.get_user_by_id(user_id, db)
        if user is None:
            raise ValueError("User not found")

        secret = self.generate_totp_secret()
        user.totp_secret = secret
        user.totp_enabled = False
        await db.flush()

        uri = self.get_totp_uri(secret, user.email)
        return {"secret": secret, "uri": uri}

    async def verify_and_enable_totp(
        self, user_id: UUID, code: str, db: AsyncSession
    ) -> bool:
        """Verify the provided TOTP *code* against the stored secret and,
        if valid, enable TOTP for the user.

        Returns ``True`` on success, ``False`` if the code is invalid.
        """
        user = await self.get_user_by_id(user_id, db)
        if user is None:
            raise ValueError("User not found")

        if not user.totp_secret:
            raise ValueError("TOTP secret not configured. Call setup_totp first.")

        if not self.verify_totp(user.totp_secret, code):
            logger.info("Invalid TOTP code during enable for user %s", user_id)
            return False

        user.totp_enabled = True
        await db.flush()
        logger.info("TOTP enabled for user %s", user_id)
        return True

    # ── User Lookups ──────────────────────────────────────────────────

    @staticmethod
    async def get_user_by_email(email: str, db: AsyncSession) -> User | None:
        """Fetch a user by email address, or ``None`` if not found."""
        stmt = select(User).where(User.email == email.lower().strip())
        result = await db.execute(stmt)
        return result.scalars().first()

    @staticmethod
    async def get_user_by_id(user_id: UUID, db: AsyncSession) -> User | None:
        """Fetch a user by primary key UUID, or ``None`` if not found."""
        stmt = select(User).where(User.id == user_id)
        result = await db.execute(stmt)
        return result.scalars().first()


# Module-level singleton for convenience (used by route handlers)
auth_service = AuthService()
