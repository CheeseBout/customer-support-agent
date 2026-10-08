from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from support_agent.core.settings import ReturnRule
from support_agent.rules.eligibility import check_return_eligibility

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
RULE = ReturnRule(
    window_days=7, allowed_order_statuses=["delivered"], excluded_categories=["gift_card"]
)


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def order(**over: Any) -> dict[str, Any]:
    base = {
        "id": "1234",
        "customer_id": "u_100",
        "status": "delivered",
        "created_at": "2026-10-05T09:00:00+07:00",
        "delivered_at": "2026-10-07T10:00:00+07:00",
    }
    return base | over


ITEMS = [
    {"sku": "EAR-BT20", "product_name": "Earbuds", "quantity": 1, "unit_price": 350_000},
    {"sku": "CASE-X100", "product_name": "Case", "quantity": 2, "unit_price": 150_000},
]
PRODUCTS = {"EAR-BT20": {"category": "accessories"}, "CASE-X100": {"category": "accessories"}}


def check(*, now: str = "2026-10-10T12:00:00+07:00", **over: Any):
    args: dict[str, Any] = dict(
        order=order(),
        items=ITEMS,
        products=PRODUCTS,
        existing_requests=[],
        requester_id="u_100",
        rule=RULE,
        now=dt(now),
        tz=TZ,
    )
    return check_return_eligibility(**(args | over))


def test_eligible_order_reports_deadline_amount_and_days_left():
    r = check()
    assert r.eligible and r.reasons == []
    assert r.deadline == "2026-10-14T00:00:00+07:00"  # delivery day counts as day 1
    assert r.days_remaining == 4  # 3.5 days left, rounded up
    assert r.refundable_amount == 350_000 + 2 * 150_000
    assert all(c.passed for c in r.checks) and {c.name for c in r.checks} >= {
        "ownership",
        "status",
        "window",
    }


@pytest.mark.parametrize(
    "now,eligible",
    [
        ("2026-10-13T23:59:59+07:00", True),  # last second of the 7th day
        ("2026-10-14T00:00:00+07:00", False),  # window closed at local midnight
        ("2026-10-07T10:00:00+07:00", True),  # delivery moment
    ],
)
def test_window_boundary(now: str, eligible: bool):
    r = check(now=now)
    assert r.eligible is eligible
    if not eligible:
        assert r.reasons == ["WINDOW_EXPIRED"] and r.days_remaining == 0


def test_window_uses_business_timezone_not_utc():
    # 18:00Z on the 6th is 01:00 on the 7th in Vietnam, so day 1 is the 7th and the deadline
    # is local midnight at the start of the 14th (= 17:00Z on the 13th).
    delivered = order(delivered_at="2026-10-06T18:00:00Z")
    last_minute = check(order=delivered, now="2026-10-13T16:59:00Z")  # 23:59 local on the 13th
    assert last_minute.eligible and last_minute.deadline == "2026-10-14T00:00:00+07:00"
    closed = check(order=delivered, now="2026-10-13T17:00:00Z")  # 00:00 local on the 14th
    assert not closed.eligible and closed.reasons == ["WINDOW_EXPIRED"]


def test_naive_timestamps_are_read_in_business_timezone():
    r = check(order=order(delivered_at="2026-10-07T10:00:00"))
    assert r.deadline == "2026-10-14T00:00:00+07:00"


def test_window_basis_can_be_order_creation_date():
    rule = RULE.model_copy(update={"window_basis": "created_at"})
    r = check(rule=rule, now="2026-10-12T12:00:00+07:00")  # created 5 Oct -> deadline 12 Oct
    assert not r.eligible and r.reasons == ["WINDOW_EXPIRED"]


def test_custom_window_length():
    rule = RULE.model_copy(update={"window_days": 30})
    assert check(rule=rule, now="2026-11-01T12:00:00+07:00").eligible


@pytest.mark.parametrize("status", ["processing", "shipping", "cancelled"])
def test_status_must_be_allowed(status: str):
    r = check(order=order(status=status))
    assert not r.eligible and "STATUS_NOT_ALLOWED" in r.reasons


def test_undelivered_order_reports_both_reasons():
    r = check(order=order(status="shipping", delivered_at=None))
    assert set(r.reasons) == {"STATUS_NOT_ALLOWED", "WINDOW_EXPIRED"}
    assert r.deadline is None and r.days_remaining is None


def test_ownership_is_required_but_staff_may_check_any_order():
    r = check(requester_id="u_999")
    assert not r.eligible and r.reasons == ["NOT_OWNER"]
    assert check(requester_id="s_1", requester_is_staff=True).eligible


def test_excluded_category_blocks_the_item():
    items = [{"sku": "GC-500K", "product_name": "Gift card", "quantity": 1, "unit_price": 500_000}]
    r = check(items=items, products={"GC-500K": {"category": "Gift_Card"}})  # case-insensitive
    assert not r.eligible and r.reasons == ["CATEGORY_EXCLUDED"]


def test_mixed_order_is_eligible_for_the_returnable_items_only():
    items = [
        *ITEMS[:1],
        {"sku": "GC-500K", "product_name": "Gift card", "quantity": 1, "unit_price": 500_000},
    ]
    r = check(items=items, products=PRODUCTS | {"GC-500K": {"category": "gift_card"}})
    assert r.eligible and r.refundable_amount == 350_000
    assert [i["sku"] for i in r.eligible_items] == ["EAR-BT20"]


@pytest.mark.parametrize("status", ["pending", "approved", "PENDING"])
def test_active_request_for_same_item_blocks(status: str):
    r = check(existing_requests=[{"status": status, "sku": "EAR-BT20"}], items=ITEMS[:1])
    assert not r.eligible and r.reasons == ["ALREADY_REQUESTED"]


def test_rejected_or_cancelled_requests_do_not_block():
    reqs = [{"status": "rejected", "sku": "EAR-BT20"}, {"status": "cancelled"}]
    assert check(existing_requests=reqs).eligible


def test_request_for_one_item_leaves_the_other_returnable():
    r = check(existing_requests=[{"status": "pending", "sku": "EAR-BT20"}])
    assert r.eligible and r.refundable_amount == 300_000


def test_order_level_request_blocks_everything():
    r = check(existing_requests=[{"status": "approved"}])
    assert not r.eligible and r.reasons == ["ALREADY_REQUESTED"]


def test_specific_sku_scope():
    r = check(sku="CASE-X100")
    assert r.eligible and r.refundable_amount == 300_000
    missing = check(sku="NOPE-1")
    assert not missing.eligible and missing.reasons == ["ITEM_NOT_FOUND"]


def test_excluded_and_already_requested_are_both_reported():
    items = [
        {"sku": "GC-500K", "product_name": "Gift card", "quantity": 1, "unit_price": 500_000},
        {"sku": "EAR-BT20", "product_name": "Earbuds", "quantity": 1, "unit_price": 350_000},
    ]
    r = check(
        items=items,
        products={"GC-500K": {"category": "gift_card"}, "EAR-BT20": {"category": "accessories"}},
        existing_requests=[{"status": "pending", "sku": "EAR-BT20"}],
    )
    assert not r.eligible and set(r.reasons) == {"CATEGORY_EXCLUDED", "ALREADY_REQUESTED"}


def test_order_without_items_is_not_eligible_and_says_why():
    r = check(items=[])
    assert not r.eligible and "ITEM_NOT_FOUND" in r.reasons


def test_unknown_category_does_not_block():
    assert check(products={}).eligible
