"""Authenticated caller identity. Never produced from LLM output."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Literal

from pydantic import BaseModel, ConfigDict

Role = Literal["customer", "staff"]


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    role: Role = "customer"


class InvalidPrincipal(Exception):
    """The signed principal is missing, malformed, expired or forged."""


def hash_user_id(user_id: str) -> str:
    """Stable short hash used in logs and audit trails instead of the raw id."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def session_namespace(user_id: str) -> str:
    """Prefix for checkpoint thread ids: a session id alone can never reach another user's data."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]


def _digest(secret: bytes, body: str) -> str:
    return hmac.new(secret, body.encode("utf-8"), hashlib.sha256).hexdigest()


def sign_principal(
    principal: Principal, secret: bytes, *, ttl_seconds: int = 300
) -> dict[str, str]:
    """Return the `_meta` entry that carries the principal across the MCP boundary."""
    body = json.dumps(
        {
            "user_id": principal.user_id,
            "role": principal.role,
            "exp": int(time.time()) + ttl_seconds,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"body": body, "sig": _digest(secret, body)}


def verify_principal(token: object, secret: bytes) -> Principal:
    if not isinstance(token, dict) or "body" not in token or "sig" not in token:
        raise InvalidPrincipal("missing principal")
    body, sig = token["body"], token["sig"]
    if not isinstance(body, str) or not isinstance(sig, str):
        raise InvalidPrincipal("malformed principal")
    if not hmac.compare_digest(_digest(secret, body), sig):
        raise InvalidPrincipal("bad signature")
    try:
        data = json.loads(body)
        expires = int(data["exp"])
        principal = Principal(user_id=str(data["user_id"]), role=data["role"])
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidPrincipal("malformed principal") from exc
    if expires < time.time():
        raise InvalidPrincipal("expired principal")
    return principal
