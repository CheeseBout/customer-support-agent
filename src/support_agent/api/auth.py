"""JWT authentication (SPEC 14.1, FR-301..304).

The shop's own system issues the tokens. We only verify them: the algorithm comes from our
configuration, never from the token header, so a token cannot pick a weaker one.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import jwt
from jwt import PyJWKClient

from support_agent.core.principal import Principal
from support_agent.core.settings import Settings

log = logging.getLogger(__name__)

MIN_SECRET_BYTES = 32


class AuthConfigError(RuntimeError):
    """The JWT settings cannot produce a working verifier."""


class InvalidToken(Exception):
    """The token is missing a claim, expired, forged or meant for someone else."""


class JwtVerifier:
    def __init__(self, settings: Settings) -> None:
        self.algorithm = settings.jwt_algorithm
        self.audience = settings.jwt_audience or None
        self.issuer = settings.jwt_issuer or None
        self.customer_claim = settings.jwt_customer_claim
        self.role_claim = settings.jwt_role_claim
        self._secret: str | None = None
        self._jwks: PyJWKClient | None = None

        if self.algorithm == "HS256":
            secret = settings.jwt_secret.get_secret_value() if settings.jwt_secret else ""
            if not secret:
                raise AuthConfigError("JWT_SECRET is required for HS256.")
            if len(secret.encode()) < MIN_SECRET_BYTES:
                log.warning(
                    "JWT_SECRET is shorter than %d bytes: use a longer one", MIN_SECRET_BYTES
                )
            self._secret = secret
        else:
            if not settings.jwt_jwks_url:
                raise AuthConfigError("JWT_JWKS_URL is required for RS256.")
            self._jwks = PyJWKClient(settings.jwt_jwks_url, cache_keys=True)

    async def verify(self, token: str) -> Principal:
        """Return the caller's identity, or raise `InvalidToken`."""
        try:
            key: Any
            if self._jwks is not None:
                # PyJWKClient does blocking HTTP (cached after the first call).
                signing = await asyncio.to_thread(self._jwks.get_signing_key_from_jwt, token)
                key = signing.key
            else:
                key = self._secret
            claims = jwt.decode(
                token,
                key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "require": ["exp", self.customer_claim],
                    "verify_aud": self.audience is not None,
                    "verify_iss": self.issuer is not None,
                },
            )
        except jwt.PyJWTError as exc:
            raise InvalidToken(type(exc).__name__) from exc

        user_id = claims.get(self.customer_claim)
        role = claims.get(self.role_claim)
        if not isinstance(user_id, str) or not user_id.strip():
            raise InvalidToken("bad customer claim")
        if role not in ("customer", "staff"):
            raise InvalidToken("bad role claim")
        return Principal(user_id=user_id.strip(), role=role)
