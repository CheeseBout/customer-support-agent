"""Drafts: requests the agent prepares and staff approve (SPEC 10).

The agent never writes to the shop's own tables. It writes a *draft* to a table it owns,
`support_drafts`, and a staff member decides what happens next.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DraftType = Literal["order", "refund", "return", "warranty"]
DraftStatus = Literal["pending", "approved", "rejected", "cancelled"]
ReasonCode = Literal[
    "defective", "wrong_item", "not_as_described", "damaged_in_transit", "changed_mind", "other"
]

# SPEC 10.3: pending -> approved | rejected | cancelled, never backwards.
TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"approved", "rejected", "cancelled"}),
    "approved": frozenset(),
    "rejected": frozenset(),
    "cancelled": frozenset(),
}
ACTIVE_STATUSES = ("pending", "approved")  # requests that still count against the customer


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OrderLine(_Payload):
    sku: str
    name: str
    qty: int = Field(ge=1)
    unit_price: int | float


class OrderPayload(_Payload):
    items: list[OrderLine] = Field(min_length=1)
    shipping_address: str = Field(min_length=5)
    payment_method: str = Field(min_length=2)
    total: int | float
    currency: str = "VND"


class RequestedItem(_Payload):
    sku: str
    qty: int = Field(ge=1)


class ReturnPayload(_Payload):
    """Used by both `return` and `refund` drafts."""

    order_id: str
    items: list[RequestedItem] = Field(min_length=1)
    reason_code: ReasonCode
    reason_text: str = ""
    refundable_amount: int | float
    eligibility_snapshot: dict[str, Any] = Field(
        default_factory=dict
    )  # the rule result at the time


class WarrantyPayload(_Payload):
    order_id: str
    sku: str
    issue_description: str = Field(min_length=3)
    warranty_expires_at: str | None = None


PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "order": OrderPayload,
    "refund": ReturnPayload,
    "return": ReturnPayload,
    "warranty": WarrantyPayload,
}


class Draft(BaseModel):
    id: str
    type: DraftType
    customer_id: str
    session_id: str
    status: DraftStatus = "pending"
    payload: dict[str, Any]
    idempotency_key: str
    priority_review: bool = False
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    created_at: datetime
    updated_at: datetime

    @property
    def order_id(self) -> str | None:
        value = self.payload.get("order_id")
        return str(value) if value is not None else None

    def sku_set(self) -> set[str]:
        """SKUs the request covers (empty = the whole order)."""
        if self.type == "warranty":
            return {str(self.payload["sku"])} if self.payload.get("sku") else set()
        return {str(i["sku"]) for i in self.payload.get("items", [])}


def idempotency_key(
    customer_id: str, session_id: str, draft_type: str, payload: dict[str, Any]
) -> str:
    """`hash(session_id, draft_type, normalised payload)` (SPEC 10.4), plus the customer.

    The customer is part of the key because `session_id` is chosen by the client: without it,
    two customers who happened to send the same session id and request would share one draft.
    The eligibility snapshot is left out: it holds clock-dependent values (days remaining), and
    a retry a minute later must still map to the same draft.
    """
    stable = {k: v for k, v in payload.items() if k != "eligibility_snapshot"}
    body = json.dumps(
        [customer_id, session_id, draft_type, stable],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()
