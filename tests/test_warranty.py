from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from support_agent.core.settings import RefundRule, WarrantyRule
from support_agent.rules.warranty import (
    add_months,
    check_warranty_eligibility,
    needs_priority_review,
    warranty_months,
)

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
RULE = WarrantyRule(default_months=12, by_category={"accessories": 6, "appliances": 24})


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


ORDER = {"id": "1", "customer_id": "u_100", "delivered_at": "2026-01-15T10:00:00+07:00"}
ITEMS = [{"sku": "PHN-X100"}, {"sku": "EAR-BT20"}, {"sku": "APL-AF30"}]
PRODUCTS = {
    "PHN-X100": {"category": "electronics"},
    "EAR-BT20": {"category": "Accessories"},  # case-insensitive match
    "APL-AF30": {"category": "appliances"},
}


def check(sku: str = "PHN-X100", *, now: str = "2026-06-01T12:00:00+07:00", **over: Any):
    args: dict[str, Any] = dict(
        order=ORDER, items=ITEMS, products=PRODUCTS, sku=sku, requester_id="u_100",
        rule=RULE, now=dt(now), tz=TZ,
    )  # fmt: skip
    return check_warranty_eligibility(**(args | over))


def test_default_period_applies_to_unlisted_categories():
    r = check()
    assert r.eligible and r.warranty_months == 12 and r.reasons == []
    assert r.expires_at == "2027-01-15T10:00:00+07:00" and r.days_remaining > 200


@pytest.mark.parametrize(
    "sku,months,expires",
    [
        ("PHN-X100", 12, "2027-01-15"),
        ("EAR-BT20", 6, "2026-07-15"),  # accessories override
        ("APL-AF30", 24, "2028-01-15"),  # appliances override
    ],
)
def test_period_depends_on_the_category(sku, months, expires):
    r = check(sku)
    assert r.warranty_months == months and r.expires_at.startswith(expires)


def test_cover_ends_exactly_at_the_end_of_the_period():
    inside = check("EAR-BT20", now="2026-07-15T10:00:00+07:00")
    assert inside.eligible  # on the last instant
    after = check("EAR-BT20", now="2026-07-15T10:00:01+07:00")
    assert (
        not after.eligible and after.reasons == ["WARRANTY_EXPIRED"] and after.days_remaining == 0
    )


def test_expired_warranty_says_when_it_ended():
    r = check("EAR-BT20", now="2026-12-01T12:00:00+07:00")
    detail = next(c.detail for c in r.checks if c.name == "period")
    assert "ended on 2026-07-15" in detail


def test_an_undelivered_order_has_no_warranty_yet():
    r = check(order=ORDER | {"delivered_at": None})
    assert not r.eligible and r.reasons == ["NOT_DELIVERED"] and r.expires_at is None


def test_the_sku_must_be_on_the_order():
    r = check("GHOST-1")
    assert not r.eligible and "ITEM_NOT_FOUND" in r.reasons


def test_only_the_owner_or_staff_may_claim():
    assert check(requester_id="u_999").reasons == ["NOT_OWNER"]
    assert check(requester_id="s_1", requester_is_staff=True).eligible


def test_naive_delivery_times_are_read_in_the_business_zone():
    r = check(order=ORDER | {"delivered_at": "2026-01-15T10:00:00"})
    assert r.expires_at == "2027-01-15T10:00:00+07:00"


def test_utc_delivery_times_are_converted_to_the_business_zone():
    r = check(order=ORDER | {"delivered_at": "2026-01-15T03:00:00Z"})  # 10:00 in Vietnam
    assert r.expires_at == "2027-01-15T10:00:00+07:00"


def test_add_months_clamps_to_the_end_of_shorter_months():
    assert add_months(dt("2026-01-31T09:00:00"), 1).date().isoformat() == "2026-02-28"
    assert add_months(dt("2028-01-31T09:00:00"), 1).date().isoformat() == "2028-02-29"  # leap year
    assert add_months(dt("2026-11-30T09:00:00"), 3).date().isoformat() == "2027-02-28"
    assert add_months(dt("2026-12-15T09:00:00"), 12).date().isoformat() == "2027-12-15"
    assert add_months(dt("2026-05-10T09:00:00"), 0).date().isoformat() == "2026-05-10"


def test_warranty_months_lookup():
    assert warranty_months("APPLIANCES", RULE) == 24
    assert warranty_months(None, RULE) == 12 and warranty_months("furniture", RULE) == 12


def test_priority_review_threshold_is_strictly_above():
    rule = RefundRule(auto_review_max_amount=500_000)
    assert needs_priority_review(500_001, rule)
    assert not needs_priority_review(500_000, rule) and not needs_priority_review(1, rule)
