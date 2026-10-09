"""Turning "I want a refund / warranty / to order this" into a validated, confirmable proposal.

The model only says WHAT the customer wants: which order, which item, why. Everything that
matters is decided here, in code:

* eligibility comes from the rules engine (the MCP tools), not from the model;
* the refundable amount, the eligible items and the order total are recomputed from the shop's
  data, so a number typed by the customer or invented by the model never reaches a draft;
* a request that is already pending or approved is not proposed a second time.

Nothing is written here. A proposal only becomes a draft after the customer confirms it.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, ValidationError

from support_agent.core.capabilities import REQUEST_TYPES
from support_agent.core.principal import Principal
from support_agent.core.reasons import ReasonCode
from support_agent.core.results import ErrorCode
from support_agent.core.settings import BusinessRules
from support_agent.drafts.service import DraftService
from support_agent.rules.eligibility import without_active_requests
from support_agent.rules.warranty import needs_priority_review
from support_agent.tools.client import DomainToolClient
from support_agent.tools.langchain_tools import OrderItemArgs, ToolArgs

# What a customer may change while confirming (SPEC FR-202, "edit"). Anything else is refused:
# editing must not become a way to rewrite the amount or the order the checks were run on.
EDITABLE: dict[str, frozenset[str]] = {
    "refund": frozenset({"reason_code", "reason_text"}),
    "return": frozenset({"reason_code", "reason_text"}),
    "warranty": frozenset({"issue_description"}),
    "order": frozenset({"shipping_address", "payment_method", "items"}),
    "handoff": frozenset({"reason_text", "contact"}),
}
MAX_EDITS = 3


class ProposeDraftArgs(ToolArgs):
    """What the model may say. No amount, no price, no identity: those are not its to give."""

    draft_type: Literal["refund", "return", "warranty", "order", "handoff"]
    order_id: str | None = Field(default=None, description="Needed for refund, return, warranty")
    sku: str | None = Field(default=None, description="The product concerned (needed for warranty)")
    items: list[OrderItemArgs] | None = Field(
        default=None, description="Order: what to buy. Refund/return: only part of the order"
    )
    reason_code: ReasonCode | None = None
    reason_text: str = Field(default="", max_length=500)
    issue_description: str | None = Field(default=None, max_length=1000)
    shipping_address: str | None = None
    payment_method: str | None = None
    contact: str | None = Field(
        default=None,
        max_length=200,
        description="Handoff only: an email or phone the customer gave for the follow-up",
    )


class ProposalError(Exception):
    """The request cannot be proposed. `code` is a tool error code the model can explain."""

    def __init__(
        self, code: ErrorCode, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass
class Proposal:
    draft_type: str
    payload: dict[str, Any]  # exactly what will be stored if the customer confirms
    summary: dict[str, Any]  # what the customer is shown
    priority_review: bool
    args: dict[str, Any] = field(default_factory=dict)  # the validated request, kept for edits


class DraftProposer:
    def __init__(
        self,
        client: DomainToolClient,
        drafts: DraftService,
        rules: BusinessRules,
        *,
        request_types: Collection[str] = REQUEST_TYPES,
    ) -> None:
        self.client = client
        self.drafts = drafts
        self.rules = rules
        self.request_types = frozenset(request_types)  # the kinds this shop takes through chat

    async def propose(self, args: ProposeDraftArgs, principal: Principal) -> Proposal:
        if args.draft_type not in self.request_types:
            raise ProposalError(
                "NOT_SUPPORTED",
                f"This shop does not take {args.draft_type} requests through the assistant; "
                "the customer should contact the shop directly.",
            )
        builders = {
            "refund": self._return_or_refund,
            "return": self._return_or_refund,
            "warranty": self._warranty,
            "order": self._order,
            "handoff": self._handoff,
        }
        proposal = await builders[args.draft_type](args, principal)
        proposal.args = args.model_dump(mode="json", exclude_none=True)
        return proposal

    async def edited(
        self, proposal: Proposal, edits: dict[str, Any], principal: Principal
    ) -> Proposal:
        """Apply the customer's corrections, then run every check again from scratch."""
        allowed = EDITABLE[proposal.draft_type]
        refused = sorted(set(edits) - allowed)
        if refused:
            raise ProposalError(
                "INVALID_ARGUMENT",
                f"Only {sorted(allowed)} can be changed while confirming; not {refused}.",
            )
        try:
            args = ProposeDraftArgs.model_validate({**proposal.args, **edits})
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            raise ProposalError("INVALID_ARGUMENT", f"Invalid edit: {fields}.") from exc
        return await self.propose(args, principal)

    # --- refund / return -------------------------------------------------------------
    async def _return_or_refund(self, args: ProposeDraftArgs, principal: Principal) -> Proposal:
        if not args.order_id:
            raise ProposalError("INVALID_ARGUMENT", "order_id is required for a refund or return.")
        if args.reason_code is None:
            raise ProposalError(
                "INVALID_ARGUMENT", "reason_code is required for a refund or return."
            )

        call_args = {
            "order_id": args.order_id,
            **({"sku": args.sku} if args.sku else {}),
            **({"reason": args.reason_code} if args.reason_code else {}),
        }
        result = await self.client.call("check_return_eligibility", call_args, principal)
        if not result.ok or not isinstance(result.data, dict):
            assert result.error is not None
            raise ProposalError(result.error.code, result.error.message)

        existing = await self.drafts.existing_requests(principal.user_id, args.order_id)
        verdict = without_active_requests(result.data, existing)
        if not verdict.get("eligible"):
            raise ProposalError(
                "NOT_ELIGIBLE",
                "This order cannot be returned or refunded.",
                {
                    "reasons": verdict.get("reasons", []),
                    "deadline": verdict.get("deadline"),
                    "checks": verdict.get("checks", []),
                    **({"order_items": verdict["order_items"]} if "order_items" in verdict else {}),
                },
            )

        eligible = {str(i["sku"]): i for i in verdict["eligible_items"]}
        chosen: dict[str, int] = {}
        if args.items:
            for line in args.items:
                item = eligible.get(line.sku)
                if item is None:
                    raise ProposalError(
                        "INVALID_ARGUMENT", f"{line.sku!r} is not eligible on this order."
                    )
                if line.qty > int(item["quantity"]):
                    raise ProposalError(
                        "INVALID_ARGUMENT",
                        f"Only {item['quantity']} of {line.sku!r} can be returned.",
                    )
                chosen[line.sku] = chosen.get(line.sku, 0) + line.qty
        else:
            chosen = {sku: int(i["quantity"]) for sku, i in eligible.items()}

        amount = sum(eligible[sku]["unit_price"] * qty for sku, qty in chosen.items())
        reason = args.reason_text.strip() or args.reason_code
        payload = {
            "order_id": str(verdict["order_id"]),
            "items": [{"sku": sku, "qty": qty} for sku, qty in chosen.items()],
            "reason_code": args.reason_code,
            "reason_text": args.reason_text.strip(),
            "refundable_amount": amount,
            "eligibility_snapshot": {
                "eligible": True,
                "deadline": verdict.get("deadline"),
                "days_remaining": verdict.get("days_remaining"),
                "checks": verdict.get("checks", []),
            },
        }
        summary = {
            "order_id": payload["order_id"],
            "items": [
                {"sku": sku, "name": eligible[sku].get("product_name"), "qty": qty}
                for sku, qty in chosen.items()
            ],
            "amount": amount,
            "currency": self.rules.refund.currency,
            "reason": reason,
        }
        return Proposal(
            draft_type=args.draft_type,
            payload=payload,
            summary=summary,
            priority_review=args.draft_type == "refund"
            and needs_priority_review(amount, self.rules.refund),
        )

    # --- handoff to a person ---------------------------------------------------------
    async def _handoff(self, args: ProposeDraftArgs, principal: Principal) -> Proposal:
        reason = (args.reason_text or "").strip()
        if len(reason) < 3:
            raise ProposalError(
                "INVALID_ARGUMENT",
                "reason_text is required: what the customer needs a person for, in their words.",
            )
        if args.order_id:  # only an order the customer owns may be attached for staff
            found = await self.client.call("get_order", {"order_id": args.order_id}, principal)
            if not found.ok:
                assert found.error is not None
                raise ProposalError(found.error.code, found.error.message)
        contact = (args.contact or "").strip()
        payload = {
            "reason": reason,
            "order_id": args.order_id,
            "contact": contact,
        }
        summary = {
            "reason": reason,
            **({"order_id": args.order_id} if args.order_id else {}),
            **({"contact": contact} if contact else {}),
        }
        return Proposal("handoff", payload, summary, priority_review=False)

    # --- warranty --------------------------------------------------------------------
    async def _warranty(self, args: ProposeDraftArgs, principal: Principal) -> Proposal:
        if not args.order_id or not args.sku:
            raise ProposalError(
                "INVALID_ARGUMENT", "order_id and sku are required for a warranty claim."
            )
        issue = (args.issue_description or "").strip()
        if len(issue) < 3:
            raise ProposalError("INVALID_ARGUMENT", "Describe the problem (issue_description).")

        result = await self.client.call(
            "check_warranty_eligibility", {"order_id": args.order_id, "sku": args.sku}, principal
        )
        if not result.ok or not isinstance(result.data, dict):
            assert result.error is not None
            raise ProposalError(result.error.code, result.error.message)
        if not result.data.get("eligible"):
            raise ProposalError(
                "NOT_ELIGIBLE",
                "This product is not covered by warranty.",
                {
                    "reasons": result.data.get("reasons", []),
                    "expires_at": result.data.get("expires_at"),
                    **(
                        {"order_items": result.data["order_items"]}
                        if "order_items" in result.data
                        else {}
                    ),
                },
            )
        payload = {
            "order_id": str(result.data["order_id"]),
            "sku": args.sku,
            "issue_description": issue,
            "warranty_expires_at": result.data.get("expires_at"),
        }
        summary = {
            "order_id": payload["order_id"],
            "sku": args.sku,
            "issue": issue,
            "warranty_expires_at": payload["warranty_expires_at"],
        }
        return Proposal("warranty", payload, summary, priority_review=False)

    # --- order -----------------------------------------------------------------------
    async def _order(self, args: ProposeDraftArgs, principal: Principal) -> Proposal:
        if not args.items:
            raise ProposalError("INVALID_ARGUMENT", "items are required to place an order.")
        if not args.shipping_address or not args.payment_method:
            raise ProposalError(
                "INVALID_ARGUMENT", "shipping_address and payment_method are required for an order."
            )
        result = await self.client.call(
            "prepare_order_draft",
            {
                "items": [{"sku": i.sku, "qty": i.qty} for i in args.items],
                "shipping_address": args.shipping_address,
                "payment_method": args.payment_method,
            },
            principal,
        )
        if not result.ok or not isinstance(result.data, dict):
            assert result.error is not None
            raise ProposalError(result.error.code, result.error.message)
        payload = result.data  # priced by the database: this is what will be stored
        summary = {
            "items": [
                {"sku": i["sku"], "name": i["name"], "qty": i["qty"], "unit_price": i["unit_price"]}
                for i in payload["items"]
            ],
            "total": payload["total"],
            "currency": payload["currency"],
            "shipping_address": payload["shipping_address"],
            "payment_method": payload["payment_method"],
        }
        return Proposal("order", payload, summary, priority_review=False)
