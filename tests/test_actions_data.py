"""The read-only tools behind the after-sales flows: warranty, comparison, order pricing."""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.mcp_db.service import BusinessService, OrderLineRequest
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, STAFF
from tests.test_actions_helpers import lines

# --- warranty --------------------------------------------------------------------------------------


async def test_warranty_is_active_for_a_recently_delivered_accessory(service: BusinessService):
    r = await service.check_warranty_eligibility(ALICE, "1234", "EAR-BT20")
    assert r.ok and r.data["eligible"] and r.data["warranty_months"] == 6  # accessories: 6 months
    assert r.data["order_id"] == "1234" and r.data["days_remaining"] > 150


async def test_warranty_length_follows_the_category_of_each_product(service: BusinessService):
    appliance = await service.check_warranty_eligibility(BOB, "2002", "APL-AF30")
    assert (
        appliance.data["warranty_months"] == 24 and not appliance.data["eligible"]
    )  # not delivered
    assert appliance.data["reasons"] == ["NOT_DELIVERED"]
    phone = await service.check_warranty_eligibility(BOB, "2001", "PHN-X100")
    assert phone.data["eligible"] and phone.data["warranty_months"] == 12  # default


async def test_warranty_for_a_sku_not_on_the_order(service: BusinessService):
    r = await service.check_warranty_eligibility(ALICE, "1234", "PHN-X100")
    assert r.ok and not r.data["eligible"] and r.data["reasons"] == ["ITEM_NOT_FOUND"]


async def test_unknown_sku_lists_the_real_lines_of_the_order(service: BusinessService):
    """The model must be able to correct an invented SKU from the answer alone."""
    warranty = await service.check_warranty_eligibility(ALICE, "1234", "EARBUDS-01")
    returns = await service.check_return_eligibility(ALICE, "1234", "EARBUDS-01")
    for r in (warranty, returns):
        assert r.data["reasons"] == ["ITEM_NOT_FOUND"]
        assert [i["sku"] for i in r.data["order_items"]] == ["EAR-BT20"]
    ok = await service.check_warranty_eligibility(ALICE, "1234", "EAR-BT20")
    assert "order_items" not in ok.data


async def test_warranty_for_someone_elses_order_is_not_found(service: BusinessService):
    foreign = await service.check_warranty_eligibility(ALICE, "2001", "PHN-X100")
    ghost = await service.check_warranty_eligibility(ALICE, "9999", "PHN-X100")
    assert foreign.error.code == ghost.error.code == "NOT_FOUND"


async def test_warranty_expiry_is_computed_from_the_delivery_date(
    service: BusinessService, monkeypatch
):
    from datetime import UTC, datetime

    monkeypatch.setattr(service, "_clock", lambda: datetime(2030, 1, 1, tzinfo=UTC))
    r = await service.check_warranty_eligibility(ALICE, "1234", "EAR-BT20")
    assert not r.data["eligible"] and r.data["reasons"] == ["WARRANTY_EXPIRED"]


# --- comparing products ------------------------------------------------------------------------------


async def test_compare_lines_up_every_specification(service: BusinessService):
    r = await service.compare_products(ALICE, ["PHN-X100", "LAP-PRO14"])
    assert r.ok and r.data["skus"] == ["PHN-X100", "LAP-PRO14"] and r.data["not_found"] == []
    table = {row["attribute"]: row["values"] for row in r.data["rows"]}
    assert table["price"] == {"PHN-X100": 7_990_000, "LAP-PRO14": 18_900_000}
    assert table["battery_mah"] == {"PHN-X100": 5000, "LAP-PRO14": None}  # only one has it
    assert table["ram_gb"] == {"PHN-X100": None, "LAP-PRO14": 16}
    assert table["availability"] == {"PHN-X100": "in_stock", "LAP-PRO14": "low_stock"}  # no counts
    assert table["name"]["PHN-X100"] == "Điện thoại Nova X100"


async def test_compare_reveals_quantities_only_when_allowed(service: BusinessService):
    staff = await service.compare_products(STAFF, ["PHN-X100", "LAP-PRO14"])
    table = {row["attribute"]: row["values"] for row in staff.data["rows"]}
    assert table["availability"]["LAP-PRO14"] == "low_stock (3)"


@pytest.mark.parametrize(
    "skus",
    [["PHN-X100"], [], ["A-1", "B-2", "C-3", "D-4", "E-5"], ["PHN-X100", "PHN-X100"], ["", "  "]],
)
async def test_compare_needs_two_to_four_distinct_products(service: BusinessService, skus):
    assert (await service.compare_products(ALICE, skus)).error.code == "INVALID_ARGUMENT"


async def test_compare_reports_unknown_products_but_still_compares_the_rest(
    service: BusinessService,
):
    r = await service.compare_products(ALICE, ["PHN-X100", "GHOST-1", "LAP-AIR13"])
    assert (
        r.ok and r.data["skus"] == ["PHN-X100", "LAP-AIR13"] and r.data["not_found"] == ["GHOST-1"]
    )
    none = await service.compare_products(ALICE, ["GHOST-1", "GHOST-2", "PHN-X100"])
    assert none.error.code == "NOT_FOUND" and "GHOST-1" in none.error.message


async def test_compare_skips_inactive_products(service: BusinessService, adapter):
    engine = create_async_engine(str(adapter.engine.url))
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE products SET is_active = 0 WHERE sku = 'LAP-PRO14'"))
    await engine.dispose()
    r = await service.compare_products(ALICE, ["PHN-X100", "LAP-PRO14", "LAP-AIR13"])
    assert r.data["skus"] == ["PHN-X100", "LAP-AIR13"] and r.data["not_found"] == ["LAP-PRO14"]


# --- pricing an order ----------------------------------------------------------------------------------

ADDRESS = "12 Le Loi, District 1, Ho Chi Minh City"


async def test_order_is_priced_from_the_database(service: BusinessService):
    r = await service.prepare_order_draft(
        ALICE, lines(("EAR-BT20", 2), ("CASE-X100", 1)), ADDRESS, "cod"
    )
    assert r.ok
    assert r.data["total"] == 2 * 350_000 + 150_000 and r.data["currency"] == "VND"
    assert [(i["sku"], i["qty"], i["unit_price"]) for i in r.data["items"]] == [
        ("EAR-BT20", 2, 350_000), ("CASE-X100", 1, 150_000),
    ]  # fmt: skip
    assert r.data["payment_method"] == "cod" and r.data["shipping_address"] == ADDRESS


async def test_repeated_skus_are_merged_and_the_address_is_tidied(service: BusinessService):
    r = await service.prepare_order_draft(
        ALICE,
        lines(("EAR-BT20", 1), ("EAR-BT20", 2)),
        "  12   Le Loi,\n District 1 ",
        "Bank Transfer",
    )
    assert r.data["items"][0]["qty"] == 3 and r.data["total"] == 1_050_000
    assert r.data["shipping_address"] == "12 Le Loi, District 1"
    assert r.data["payment_method"] == "bank_transfer"  # normalised


@pytest.mark.parametrize(
    "items,address,method,code",
    [
        ([("EAR-BT20", 21)], ADDRESS, "cod", "LIMIT_EXCEEDED"),  # > 20 of one product
        ([], ADDRESS, "cod", "INVALID_ARGUMENT"),
        ([("EAR-BT20", 1)], "x", "cod", "INVALID_ARGUMENT"),  # no real address
        ([("EAR-BT20", 1)], ADDRESS, "bitcoin", "NOT_ELIGIBLE"),
        ([("GHOST-1", 1)], ADDRESS, "cod", "NOT_FOUND"),
        ([("CHG-65W", 1)], ADDRESS, "cod", "OUT_OF_STOCK"),  # 0 on hand
        ([("LAP-PRO14", 4)], ADDRESS, "cod", "OUT_OF_STOCK"),  # only 3
    ],
)
async def test_invalid_orders_are_refused_with_the_right_code(
    service, items, address, method, code
):
    r = await service.prepare_order_draft(ALICE, lines(*items), address, method)
    assert not r.ok and r.error.code == code


async def test_an_order_may_have_at_most_ten_different_products(service: BusinessService):
    skus = ["PHN-X100", "LAP-PRO14", "LAP-AIR13", "EAR-BT20", "CHG-65W", "CASE-X100",
            "MOU-M10", "KEY-K75", "APL-AF30", "APL-AP20", "SPK-BM5"]  # fmt: skip
    r = await service.prepare_order_draft(ALICE, lines(*[(s, 1) for s in skus]), ADDRESS, "card")
    assert r.error.code == "LIMIT_EXCEEDED" and "at most 10" in r.error.message


async def test_cash_on_delivery_is_capped_but_other_methods_are_not(service: BusinessService):
    big = lines(("PHN-X100", 1))  # 7,990,000 > the 5,000,000 cap in the payment policy
    cod = await service.prepare_order_draft(ALICE, big, ADDRESS, "cod")
    assert cod.error.code == "NOT_ELIGIBLE" and "5000000" in cod.error.message
    assert (await service.prepare_order_draft(ALICE, big, ADDRESS, "card")).ok
    small = await service.prepare_order_draft(ALICE, lines(("EAR-BT20", 1)), ADDRESS, "cod")
    assert small.ok


async def test_stock_shortage_hides_quantities_from_customers_but_not_staff(
    service: BusinessService,
):
    customer = await service.prepare_order_draft(ALICE, lines(("LAP-PRO14", 5)), ADDRESS, "card")
    assert customer.error.code == "OUT_OF_STOCK" and "3" not in customer.error.message
    staff = await service.prepare_order_draft(STAFF, lines(("LAP-PRO14", 5)), ADDRESS, "card")
    assert "'LAP-PRO14': 3" in staff.error.message


async def test_inactive_products_cannot_be_ordered(service: BusinessService, adapter):
    engine = create_async_engine(str(adapter.engine.url))
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE products SET is_active = 0 WHERE sku = 'EAR-BT20'"))
    await engine.dispose()
    r = await service.prepare_order_draft(ALICE, lines(("EAR-BT20", 1)), ADDRESS, "cod")
    assert r.error.code == "NOT_FOUND"


async def test_a_negative_or_zero_quantity_never_reaches_the_service():
    with pytest.raises(ValueError):
        OrderLineRequest(sku="A-1", qty=0)
    with pytest.raises(ValueError):
        OrderLineRequest(sku="A-1", qty=-3)


# --- over MCP --------------------------------------------------------------------------------------------


async def test_the_new_tools_work_over_mcp(tool_client: DomainToolClient):
    warranty = await tool_client.call(
        "check_warranty_eligibility", {"order_id": "1234", "sku": "EAR-BT20"}, ALICE
    )
    assert warranty.ok and warranty.data["eligible"]
    compare = await tool_client.call("compare_products", {"skus": ["PHN-X100", "LAP-PRO14"]}, ALICE)
    assert compare.ok and compare.data["skus"] == ["PHN-X100", "LAP-PRO14"]
    order = await tool_client.call(
        "prepare_order_draft",
        {
            "items": [{"sku": "EAR-BT20", "qty": 2}],
            "shipping_address": ADDRESS,
            "payment_method": "momo",
        },
        ALICE,
    )
    assert order.ok and order.data["total"] == 700_000


async def test_the_new_tools_refuse_identity_arguments(tool_client: DomainToolClient):
    calls: list[tuple[str, dict[str, Any]]] = [
        ("check_warranty_eligibility", {"order_id": "2001", "sku": "PHN-X100", "customer_id": "u_101"}),
        ("compare_products", {"skus": ["PHN-X100", "LAP-PRO14"], "user_id": "u_101"}),
        ("prepare_order_draft", {"items": [{"sku": "EAR-BT20", "qty": 1}], "shipping_address": ADDRESS,
                                 "payment_method": "cod", "role": "staff"}),
    ]  # fmt: skip
    for tool, args in calls:
        assert (await tool_client.call(tool, args, ALICE)).error.code == "FORBIDDEN", tool


async def test_a_price_supplied_by_the_caller_is_rejected_not_used(tool_client: DomainToolClient):
    r = await tool_client.call(
        "prepare_order_draft",
        {"items": [{"sku": "EAR-BT20", "qty": 1, "unit_price": 1}], "shipping_address": ADDRESS, "payment_method": "cod"},
        ALICE,
    )  # fmt: skip
    assert not r.ok  # the item schema forbids extra fields, so a cheap price cannot be smuggled in
