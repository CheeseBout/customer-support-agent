"""Warranty eligibility (SPEC 9.2) and the refund review threshold (SPEC 9.4). Pure functions."""

from __future__ import annotations

import calendar
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from support_agent.core.settings import RefundRule, WarrantyRule
from support_agent.rules.eligibility import Check, _parse


class WarrantyResult(BaseModel):
    eligible: bool
    reasons: list[str] = Field(default_factory=list)
    sku: str | None = None
    warranty_months: int | None = None
    expires_at: str | None = None
    days_remaining: int | None = None
    checks: list[Check] = Field(default_factory=list)


def add_months(moment: datetime, months: int) -> datetime:
    """`moment` plus whole calendar months, clamping the day (31 Jan + 1 month = 28/29 Feb)."""
    index = moment.month - 1 + months
    year, month = moment.year + index // 12, index % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def warranty_months(category: str | None, rule: WarrantyRule) -> int:
    """Months of cover for a product category. `by_category` overrides the default."""
    wanted = (category or "").lower()
    for name, months in rule.by_category.items():
        if name.lower() == wanted:
            return months
    return rule.default_months


def check_warranty_eligibility(
    *,
    order: dict[str, Any],
    items: list[dict[str, Any]],
    products: dict[str, dict[str, Any]],
    sku: str,
    requester_id: str | None,
    rule: WarrantyRule,
    now: datetime,
    tz: ZoneInfo,
    requester_is_staff: bool = False,
) -> WarrantyResult:
    """Is `sku` from `order` still under warranty? Cover starts on the delivery date."""
    checks: list[Check] = []
    reasons: list[str] = []

    def record(name: str, passed: bool, detail: str, reason: str) -> None:
        checks.append(Check(name=name, passed=passed, detail=detail))
        if not passed and reason not in reasons:
            reasons.append(reason)

    owner_ok = requester_is_staff or str(order.get("customer_id")) == str(requester_id)
    record(
        "ownership",
        owner_ok,
        "order belongs to the requester" if owner_ok else "not the owner",
        "NOT_OWNER",
    )

    item = next((i for i in items if str(i.get("sku")) == str(sku)), None)
    record(
        "item",
        item is not None,
        f"SKU {sku!r} is on the order" if item else f"SKU {sku!r} is not on this order",
        "ITEM_NOT_FOUND",
    )

    delivered = _parse(order.get("delivered_at"), tz)
    record(
        "delivered",
        delivered is not None,
        "delivered" if delivered else "the order has no delivery date: warranty starts on delivery",
        "NOT_DELIVERED",
    )

    months = expires = remaining = None
    if item is not None:  # the length of cover is known even before delivery starts the clock
        months = warranty_months((products.get(str(sku)) or {}).get("category"), rule)
    if delivered is not None and months is not None:
        expires_dt = add_months(delivered, months)
        local_now = now.astimezone(tz)
        within = local_now <= expires_dt
        remaining = max(0, -(-int((expires_dt - local_now).total_seconds()) // 86400))
        expires = expires_dt.isoformat()
        record(
            "period",
            within,
            f"{months} months of cover until {expires_dt.date().isoformat()}"
            if within
            else f"{months}-month warranty ended on {expires_dt.date().isoformat()}",
            "WARRANTY_EXPIRED",
        )

    return WarrantyResult(
        eligible=not reasons,
        reasons=reasons,
        sku=str(sku),
        warranty_months=months,
        expires_at=expires,
        days_remaining=remaining if not reasons else (0 if "WARRANTY_EXPIRED" in reasons else None),
        checks=checks,
    )


def needs_priority_review(amount: int | float, rule: RefundRule) -> bool:
    """A refund above the configured amount is flagged for senior staff; none is auto-approved."""
    return amount > rule.auto_review_max_amount
