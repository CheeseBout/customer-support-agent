"""Request and response models (SPEC 14). They are what OpenAPI shows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from support_agent.drafts.models import Draft, DraftStatus, DraftType

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- chat -----------------------------------------------------------------------------


class ChatRequest(_Strict):
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description="Continue this conversation. Empty: the server creates one and returns its id.",
    )
    message: str = Field(min_length=1, description="The customer's message.")
    language: Literal["auto", "vi", "en"] = "auto"

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"message": "Đơn #1234 của tôi còn được trả hàng không?", "language": "auto"}
            ]
        },
    )


class Citation(BaseModel):
    source: str
    section: str


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Interrupt(BaseModel):
    """A request the customer must confirm before anything is created (SPEC 14.4)."""

    id: str
    kind: Literal["confirm_draft"] = "confirm_draft"
    draft_type: DraftType
    summary: dict[str, Any]
    priority_review: bool = False
    allowed_decisions: list[Literal["approve", "reject", "edit"]]
    editable_fields: list[str] = Field(default_factory=list)
    expires_at: datetime


class CreatedDraft(BaseModel):
    id: str
    type: DraftType
    status: DraftStatus


class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    outcome: str = Field(
        description="answered | no_info | not_found | clarify | refused | confirmation | error"
    )
    language: Literal["vi", "en"]
    route: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    interrupt: Interrupt | None = None
    drafts: list[CreatedDraft] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    trace_id: str | None = None


class ResumeRequest(_Strict):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    interrupt_id: str = Field(min_length=1, max_length=64)
    decision: Literal["approve", "reject", "edit"]
    edits: dict[str, Any] | None = Field(
        default=None, description="Changed fields, only with decision=edit."
    )


# --- sessions -------------------------------------------------------------------------


class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SessionList(BaseModel):
    sessions: list[SessionSummary]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionDetail(SessionSummary):
    messages: list[Message]
    pending_interrupt: Interrupt | None = None
    artifacts: list[str] = Field(default_factory=list)


# --- drafts ---------------------------------------------------------------------------


class DraftOut(BaseModel):
    id: str
    type: DraftType
    status: DraftStatus
    payload: dict[str, Any]
    priority_review: bool
    session_id: str
    review_note: str | None = None
    reviewed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, draft: Draft) -> DraftOut:
        """The customer's view: no staff id, no idempotency key."""
        return cls(**draft.model_dump(exclude={"customer_id", "reviewed_by", "idempotency_key"}))


class AdminDraftOut(DraftOut):
    customer_id: str
    reviewed_by: str | None = None

    @classmethod
    def of(cls, draft: Draft) -> AdminDraftOut:
        return cls(**draft.model_dump(exclude={"idempotency_key"}))


class DraftList(BaseModel):
    drafts: list[DraftOut]


class AdminDraftList(BaseModel):
    drafts: list[AdminDraftOut]
    limit: int
    offset: int


class ApproveRequest(_Strict):
    note: str | None = Field(default=None, max_length=1000)


class RejectRequest(_Strict):
    note: str = Field(min_length=1, max_length=1000, description="Shown to the customer.")


class IngestRequest(_Strict):
    full: bool = False


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    removed: list[str]
    failed: dict[str, str]
    chunks_written: int


# --- feedback, memory -----------------------------------------------------------------


class FeedbackRequest(_Strict):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)


class FactIn(_Strict):
    key: Literal["preferred_language", "product_interests"]
    value: str = Field(min_length=1, max_length=200)


class FactOut(BaseModel):
    key: str
    value: str
    created_at: datetime
    expires_at: datetime


class MemoryView(BaseModel):
    enabled: bool
    facts: list[FactOut]


# --- health ---------------------------------------------------------------------------


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]
