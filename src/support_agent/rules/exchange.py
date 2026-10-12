"""Exchange eligibility. Pure functions: no I/O, no LLM, fully unit-tested.

An exchange swaps an item the customer received for another one: a different size or colour of
the same product, or a new unit of a faulty one. It is not a return: the shop gives it its own
window (`ExchangeRule`), its own limit per order, and no money changes hands, so a return's
deadline must never be used to judge it.

The time window, the order status, the excluded categories and the per-reason or per-category
windows are decided exactly as for a return, by `check_return_eligibility`, applied to the
exchange rule. This module adds the limit of exchanges per order and the check of the item the
customer wants instead.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from support_agent.core.settings import ExchangeRule
from support_agent.rules.eligibility import Check, check_return_eligibility

# An exchange that was turned down or withdrawn does not use up the customer's allowance.
NOT_COUNTED = {"rejected", "cancelled"}
# Reasons that mean the item itself is wrong: only then may it be replaced by the same product.
FAULT_REASONS = {"defective", "wrong_item", "not_as_described", "damaged_in_transit"}


class ExchangeResult(BaseModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    deadline: str | None = None
    days_remaining: int | None = None
    eligible_items: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    # Longer windows that a reason (for example a fault) would unlock; see `ReturnRule.windows`.
    reason_windows: list[dict[str, Any]] = Field(default_factory=list)
    exchanges_used: int = 0
    exchanges_allowed: int | None = None  # None: no limit
    # Set when the customer named the item they want: whether it can be had.
    replacement: dict[str, Any] | None = None


def exchanges_used(existing: list[dict[str, Any]]) -> int:
    """How many exchanges an order already has. A row per item still counts as one exchange when
    the rows share an id."""
    counted = [r for r in existing if str(r.get("status", "")).lower() not in NOT_COUNTED]
    ids = {str(r["id"]) for r in counted if r.get("id") is not None}
    anonymous = sum(1 for r in counted if r.get("id") is None)
    return len(ids) + anonymous


def check_exchange_eligibility(
    *,
    order: dict[str, Any],
    items: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    existing_exchanges: list[dict[str, Any]],
    existing_returns: list[dict[str, Any]],
    requester_id: str | None,
    rule: ExchangeRule,
    now: datetime,
    tz: ZoneInfo,
    sku: str | None = None,
    requester_is_staff: bool = False,
    reason: str | None = None,
) -> ExchangeResult:
    """Decide whether (part of) `order` can be exchanged."""
    base = check_return_eligibility(
        order=order,
        items=items,
        products=products,
        existing_requests=existing_returns,  # an item already being returned cannot be swapped
        requester_id=requester_id,
        rule=rule,
        now=now,
        tz=tz,
        sku=sku,
        requester_is_staff=requester_is_staff,
        reason=reason,
    )
    used = exchanges_used(existing_exchanges)
    allowed = rule.max_per_order
    reasons = list(base.reasons)
    checks = list(base.checks)
    within_limit = allowed is None or used < allowed
    checks.append(
        Check(
            name="exchange_limit",
            passed=within_limit,
            detail="no limit on exchanges per order"
            if allowed is None
            else f"{used} of {allowed} exchange(s) used on this order",
        )
    )
    if not within_limit and "EXCHANGE_LIMIT_REACHED" not in reasons:
        reasons.append("EXCHANGE_LIMIT_REACHED")
    eligible = base.eligible and within_limit
    return ExchangeResult(
        eligible=eligible,
        reasons=reasons,
        deadline=base.deadline,
        days_remaining=base.days_remaining,
        eligible_items=[
            {k: v for k, v in item.items() if k != "unit_price"} for item in base.eligible_items
        ]
        if eligible
        else [],
        checks=checks,
        reason_windows=base.reason_windows,
        exchanges_used=used,
        exchanges_allowed=allowed,
    )


def with_pending_exchanges(result: dict[str, Any], pending: int) -> dict[str, Any]:
    """Count the exchanges the agent has itself taken but the shop's data does not hold yet.

    `result` is the JSON form of `ExchangeResult`. Same idea as `without_active_requests`: two
    requests made through the chat must not both get the one exchange an order is allowed.
    """
    allowed = result.get("exchanges_allowed")
    if pending <= 0:
        return result
    used = int(result.get("exchanges_used") or 0) + pending
    updated = {**result, "exchanges_used": used}
    if not result.get("eligible") or allowed is None or used < allowed:
        return updated
    updated["eligible"] = False
    updated["eligible_items"] = []
    updated["reasons"] = [*result.get("reasons", []), "EXCHANGE_LIMIT_REACHED"]
    updated["checks"] = [
        *result.get("checks", []),
        {
            "name": "exchange_limit",
            "passed": False,
            "detail": f"{used} of {allowed} exchange(s) used on this order",
        },
    ]
    return updated


def check_replacement(
    *,
    sku: str,
    new_sku: str,
    products: dict[str, dict[str, Any]],
    stock: dict[str, int] | None,
    rule: ExchangeRule,
    reason: str | None,
) -> dict[str, Any]:
    """Can the customer have `new_sku` instead of `sku`? `stock` is None when the shop has no
    stock data (availability is then unknown, not refused)."""
    problems: list[str] = []
    old, new = products.get(sku) or {}, products.get(new_sku)
    if new is None or new.get("active") is False:
        problems.append("REPLACEMENT_NOT_FOUND")
    elif new_sku == sku:
        if reason not in FAULT_REASONS:
            problems.append("REPLACEMENT_SAME_ITEM")  # swapping an item for itself needs a fault
    elif rule.same_product_only:
        group_old, group_new = old.get("group_id"), new.get("group_id")
        if group_old is None or group_new is None or str(group_old) != str(group_new):
            problems.append("REPLACEMENT_DIFFERENT_PRODUCT")
    available: bool | None = None
    if new is not None and stock is not None:
        available = stock.get(new_sku, 0) > 0
        if not available:
            problems.append("REPLACEMENT_OUT_OF_STOCK")
    return {
        "sku": new_sku,
        "name": (new or {}).get("name"),
        "options": (new or {}).get("options"),
        "available": available,
        "valid": not problems,
        "problems": problems,
    }
