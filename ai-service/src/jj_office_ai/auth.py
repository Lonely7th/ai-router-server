from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import HTTPException, Request, status

from .config import Settings

USER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,255}$")


@dataclass(frozen=True, slots=True)
class AuthContext:
    user_id: str
    plan: str | None = None
    session_id: str | None = None

    @property
    def provider_user_id(self) -> str:
        # DeepSeek user_id must not contain private information. A stable opaque hash also
        # keeps the provider-facing identifier inside its documented character set.
        return hashlib.sha256(self.user_id.encode("utf-8")).hexdigest()[:40]


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token.strip()


def _safe_user_id(value: str | None) -> str:
    if not value or not USER_ID_RE.fullmatch(value):
        raise HTTPException(status_code=401, detail="Invalid authenticated user")
    return value


def _authenticate_static(request: Request, settings: Settings) -> AuthContext:
    supplied = _bearer_token(request)
    valid = any(
        hmac.compare_digest(supplied, expected) for expected in settings.parsed_static_tokens
    )
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthContext(user_id=_safe_user_id(request.headers.get("x-user-id")))


def _authenticate_jwt(request: Request, settings: Settings) -> AuthContext:
    token = _bearer_token(request)
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_public_key,
            algorithms=["RS256"],
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return AuthContext(
        user_id=_safe_user_id(str(claims.get("sub", ""))),
        plan=str(claims["plan"]) if claims.get("plan") is not None else None,
        session_id=str(claims["sid"]) if claims.get("sid") is not None else None,
    )


async def authenticate(request: Request) -> AuthContext:
    settings: Settings = request.app.state.settings
    if settings.auth_mode == "none":
        user_id = request.headers.get("x-user-id") or "anonymous"
        return AuthContext(user_id=_safe_user_id(user_id))
    if settings.auth_mode == "static":
        return _authenticate_static(request, settings)
    return _authenticate_jwt(request, settings)
