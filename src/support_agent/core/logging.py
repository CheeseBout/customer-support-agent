"""Structured JSON logging with request context and PII masking."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from support_agent.security.pii import mask_text

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
session_id_var: ContextVar[str | None] = ContextVar("session_id", default=None)
user_hash_var: ContextVar[str | None] = ContextVar("user_hash", default=None)

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


def new_request_id() -> str:
    return uuid.uuid4().hex


def bind_context(
    *, request_id: str | None = None, session_id: str | None = None, user_hash: str | None = None
) -> None:
    if request_id is not None:
        request_id_var.set(request_id)
    if session_id is not None:
        session_id_var.set(session_id)
    if user_hash is not None:
        user_hash_var.set(user_hash)


class JsonFormatter(logging.Formatter):
    def __init__(self, *, mask_pii: bool = True) -> None:
        super().__init__()
        self._mask = mask_pii

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_text(message) if self._mask else message,
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "user_id": user_hash_var.get(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = mask_text(value) if self._mask and isinstance(value, str) else value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(
            {k: v for k, v in payload.items() if v is not None}, ensure_ascii=False, default=str
        )


def configure_logging(level: str = "INFO", *, mask_pii: bool = True) -> None:
    """Idempotent. Logs go to stderr so stdout stays clean (required by MCP stdio servers)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_support_agent", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(mask_pii=mask_pii))
    handler._support_agent = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level.upper())
    for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
