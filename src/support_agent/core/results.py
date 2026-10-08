"""Uniform tool result envelope: {ok, data?, error?{code, message}} (SPEC 8.4)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ErrorCode = Literal[
    "NOT_FOUND",
    "FORBIDDEN",
    "INVALID_ARGUMENT",
    "CONFLICT",
    "NOT_ELIGIBLE",
    "OUT_OF_STOCK",
    "LIMIT_EXCEEDED",
    "UPSTREAM_ERROR",
    "TIMEOUT",
]


class ToolError(BaseModel):
    code: ErrorCode
    message: str


class ToolResult(BaseModel):
    ok: bool
    data: Any | None = None
    error: ToolError | None = None

    @classmethod
    def success(cls, data: Any) -> ToolResult:
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, code: ErrorCode, message: str) -> ToolResult:
        return cls(ok=False, error=ToolError(code=code, message=message))

    def to_wire(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)
