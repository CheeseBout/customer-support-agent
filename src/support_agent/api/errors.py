"""One error shape for the whole API (SPEC 14.6) and the request-id middleware."""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from support_agent.core.logging import bind_context, new_request_id, request_id_var

log = logging.getLogger(__name__)

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_CODE_BY_STATUS = {
    400: "INVALID_REQUEST",
    401: "UNAUTHENTICATED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "INVALID_REQUEST",
    409: "CONFLICT",
    422: "GUARDRAIL_BLOCKED",
    429: "RATE_LIMITED",
    502: "UPSTREAM_ERROR",
    503: "NOT_CONFIGURED",
    504: "TIMEOUT",
}


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        code: str | None = None,
        message: str = "",
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code or _CODE_BY_STATUS.get(status, "INTERNAL_ERROR")
        self.message = message
        self.headers = headers or {}


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    """`{"error": {"code": ..., "message": ..., "request_id": ...}}`"""

    error: ErrorBody


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": name}
    for code, name in (
        (400, "INVALID_REQUEST"),
        (401, "UNAUTHENTICATED"),
        (403, "FORBIDDEN"),
        (404, "NOT_FOUND"),
        (409, "CONFLICT"),
        (422, "GUARDRAIL_BLOCKED"),
        (429, "RATE_LIMITED"),
        (502, "UPSTREAM_ERROR"),
        (504, "TIMEOUT"),
    )
}


def error_response(
    status: int, code: str, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=code, message=message, request_id=request_id_var.get())
    )
    return JSONResponse(body.model_dump(), status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        return error_response(400, "INVALID_REQUEST", f"Invalid request: {', '.join(fields)}.")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _CODE_BY_STATUS.get(exc.status_code, "INTERNAL_ERROR")
        return error_response(exc.status_code, code, str(exc.detail), dict(exc.headers or {}))

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return error_response(500, "INTERNAL_ERROR", "Unexpected server error.")


class RequestIdMiddleware:
    """Honour a sane `X-Request-ID`, else make one; bind it to the logs; echo it back."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        supplied = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = supplied if _REQUEST_ID.match(supplied) else new_request_id()
        bind_context(request_id=request_id)

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message["headers"] if k.lower() != b"x-request-id"]
                headers.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_id)
