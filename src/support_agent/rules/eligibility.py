"""Return/refund eligibility (SPEC 9.1). Pure functions: no I/O, no LLM, fully unit-tested.

The window is counted in calendar days of the business timezone, with the delivery day as
day 1: delivered on 7 Oct with a 7-day window -> eligible through 13 Oct, deadline 14 Oct 00:00.

A shop can give some items or some reasons a different window (`ReturnRule.windows`), for
example 30 days for a faulty item and 7 for a change of mind. Every item then has its own
deadline.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from support_agent.core.settings import ReturnRule

ACTIVE_REQUEST_STATUSES = {"pending", "approved"}


class Check(BaseModel):
    name: str
    passed: bool
    detail: str


class EligibilityResult(BaseModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    deadline: str | None = None
    days_remaining: int | None = None
    refundable_amount: int | float = 0
    eligible_items: list[dict[str, Any]] = Field(default_factory=list)
    checks: list[Check] = Field(default_factory=list)
    # Windows that would be longer for a reason the customer has not given yet, e.g. a faulty
    # item. Empty when the shop has no reason-specific windows or the reason is known.
    reason_windows: list[dict[str, Any]] = Field(default_factory=list)


def _parse(ts: Any, tz: ZoneInfo) -> datetime | None:
    if ts is None:
        return None
    dt = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=tz)).astimezone(tz)


def _line_total(item: dict[str, Any]) -> int | float:
    return (item.get("unit_price") or 0) * (item.get("quantity") or 0)


def window_days_for(rule: ReturnRule, category: str, reason: str | None) -> int:
    """The return window for an item: the first matching entry of `rule.windows`, else the default.

    An entry that names reasons never matches while the reason is unknown, so asking "can I
    return this?" gets the strict answer, and `reason_windows` says what a reason could change.
    """
    for entry in rule.windows:
        if entry.reasons and reason not in entry.reasons:
            continue
        if entry.categories and category not in {c.lower() for c in entry.categories}:
            continue
        return entry.days
    return rule.window_days


def check_return_eligibility(
    *,
    order: dict[str, Any],
    items: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    existing_requests: list[dict[str, Any]],
    requester_id: str | None,
    rule: ReturnRule,
    now: datetime,
    tz: ZoneInfo,
    sku: str | None = None,
    requester_is_staff: bool = False,
    reason: str | None = None,
) -> EligibilityResult:
    """Decide whether (part of) `order` can be returned or refunded."""
    checks: list[Check] = []
    reasons: list[str] = []

    def record(name: str, passed: bool, detail: str, reason: str) -> None:
        checks.append(Check(name=name, passed=passed, detail=detail))
        if not passed and reason not in reasons:
            reasons.append(reason)

    def category_of(item: dict[str, Any]) -> str:
        return str((products.get(str(item.get("sku")), {}) or {}).get("category") or "").lower()

    # 1. ownership
    owner_ok = requester_is_staff or str(order.get("customer_id")) == str(requester_id)
    record(
        "ownership",
        owner_ok,
        "order belongs to the requester" if owner_ok else "not the owner",
        "NOT_OWNER",
    )

    # 2. status
    status = order.get("status")
    status_ok = status in rule.allowed_order_statuses
    record(
        "status",
        status_ok,
        f"order status is {status!r}; allowed: {rule.allowed_order_statuses}",
        "STATUS_NOT_ALLOWED",
    )

    # 3. time window. Each item has its own window (category, reason); the order-level numbers
    # describe the item with the most generous one.
    scope = [i for i in items if sku is None or str(i.get("sku")) == str(sku)]
    basis_ts = _parse(order.get(rule.window_basis), tz)
    local_now = now.astimezone(tz)
    deadline: datetime | None = None
    days_remaining: int | None = None
    item_deadline: dict[int, datetime] = {}
    reason_windows: list[dict[str, Any]] = []
    if basis_ts is None:
        record("window", False, f"no {rule.window_basis} date on the order", "WINDOW_EXPIRED")
    else:
        start = basis_ts.replace(hour=0, minute=0, second=0, microsecond=0)
        days = {n: window_days_for(rule, category_of(item), reason) for n, item in enumerate(scope)}
        item_deadline = {n: start + timedelta(days=d) for n, d in days.items()}
        best_days = max(days.values(), default=rule.window_days)
        deadline = start + timedelta(days=best_days)
        in_window = local_now < deadline
        days_remaining = max(0, math.ceil((deadline - local_now).total_seconds() / 86400))
        detail = (
            f"{days_remaining} day(s) left (deadline {deadline.isoformat()})"
            if in_window
            else f"{best_days}-day window from {rule.window_basis} ended {deadline.isoformat()}"
        )
        late = sorted(
            str(scope[n].get("sku")) for n, dl in item_deadline.items() if local_now >= dl
        )
        if in_window and late:
            detail += f"; past their window: {late}"
        record("window", in_window, detail, "WINDOW_EXPIRED")
        if reason is None:
            reason_windows = _reason_windows(rule, scope, days, start, local_now, category_of)

    # 4 + 5. item-level checks
    if not scope:
        detail = f"SKU {sku!r} is not part of this order" if sku else "order has no items"
        record("item", False, detail, "ITEM_NOT_FOUND")

    excluded = {c.lower() for c in rule.excluded_categories}
    requested_skus: set[str] = set()
    order_level_block = False
    for req in existing_requests:
        if str(req.get("status", "")).lower() in ACTIVE_REQUEST_STATUSES:
            if req.get("sku"):
                requested_skus.add(str(req["sku"]))
            else:
                order_level_block = True

    candidates: list[tuple[int, dict[str, Any]]] = []
    excluded_hits: list[str] = []
    already: list[str] = []
    for n, item in enumerate(scope):
        category = category_of(item)
        if category and category in excluded:
            excluded_hits.append(str(item.get("sku")))
        elif order_level_block or str(item.get("sku")) in requested_skus:
            already.append(str(item.get("sku")))
        else:
            candidates.append((n, item))

    if scope:
        record(
            "category",
            not (excluded_hits and not candidates),
            f"excluded categories: {sorted(excluded) or 'none'}"
            + (f"; excluded SKUs: {excluded_hits}" if excluded_hits else ""),
            "CATEGORY_EXCLUDED",
        )
        record(
            "no_active_request",
            not (already and not candidates),
            "no pending/approved request for these items"
            if not already
            else f"active request already exists for: {already}",
            "ALREADY_REQUESTED",
        )

    # Items can have different windows: only those still inside their own can be returned.
    eligible_items = [
        (n, i) for n, i in candidates if n not in item_deadline or local_now < item_deadline[n]
    ]
    if candidates and not eligible_items and "WINDOW_EXPIRED" not in reasons:
        record(
            "item_window",
            False,
            "every remaining item is past its own return window",
            "WINDOW_EXPIRED",
        )
    eligible = not reasons and bool(eligible_items)

    def view(n: int, item: dict[str, Any]) -> dict[str, Any]:
        out = {k: item.get(k) for k in ("sku", "product_name", "quantity", "unit_price")}
        if rule.windows:  # items can differ, so say when each one runs out
            out["deadline"] = item_deadline[n].isoformat()
        return out

    return EligibilityResult(
        eligible=eligible,
        reasons=reasons,
        deadline=deadline.isoformat() if deadline else None,
        days_remaining=days_remaining,
        refundable_amount=sum(_line_total(i) for _, i in eligible_items) if eligible else 0,
        eligible_items=[view(n, i) for n, i in eligible_items] if eligible else [],
        checks=checks,
        reason_windows=reason_windows,
    )


def _reason_windows(
    rule: ReturnRule,
    scope: list[dict[str, Any]],
    applied: dict[int, int],
    start: datetime,
    local_now: datetime,
    category_of: Any,
) -> list[dict[str, Any]]:
    """Longer windows that a reason would unlock for the items in `scope`."""
    found: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, ...], int]] = set()
    for n, item in enumerate(scope):
        for entry in rule.windows:
            if not entry.reasons or entry.days <= applied[n]:
                continue
            if entry.categories and category_of(item) not in {c.lower() for c in entry.categories}:
                continue
            key = (tuple(entry.reasons), entry.days)
            if key in seen:
                continue
            seen.add(key)
            until = start + timedelta(days=entry.days)
            found.append(
                {
                    "reasons": list(entry.reasons),
                    "days": entry.days,
                    "deadline": until.isoformat(),
                    "still_open": local_now < until,
                }
            )
    return found


def without_active_requests(
    result: dict[str, Any], existing: list[dict[str, Any]]
) -> dict[str, Any]:
    """Drop items that already have a pending/approved request from an eligibility result.

    `result` is the JSON form of `EligibilityResult` (what the MCP tool returns). The agent
    adds its own pending drafts here, because the shop's database cannot know about them yet;
    otherwise asking twice would stack two refunds on one product (SPEC 9.1, check 5).
    """
    if not result.get("eligible"):
        return result
    active = [r for r in existing if str(r.get("status", "")).lower() in ACTIVE_REQUEST_STATUSES]
    whole_order = any(not r.get("sku") for r in active)
    blocked = {str(r["sku"]) for r in active if r.get("sku")}
    items = result.get("eligible_items", [])
    kept = [] if whole_order else [i for i in items if str(i.get("sku")) not in blocked]
    if len(kept) == len(items):
        return result

    updated = dict(result)
    updated["eligible_items"] = kept
    updated["refundable_amount"] = sum(_line_total(i) for i in kept)
    updated["checks"] = [
        *result.get("checks", []),
        {
            "name": "no_pending_request",
            "passed": bool(kept),
            "detail": "a request for the remaining items is already pending or approved"
            if not kept
            else f"items {sorted(blocked)} already have a pending or approved request",
        },
    ]
    if not kept:
        updated["eligible"] = False
        updated["reasons"] = [*result.get("reasons", []), "ALREADY_REQUESTED"]
    return updated
