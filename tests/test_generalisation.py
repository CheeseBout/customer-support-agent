"""A shop does not have to look like the demo: optional data, more statuses, its own currency."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from support_agent.agent.skills import (
    format_money,
    load_skill,
    render,
    skill_facts,
    skill_names,
)
from support_agent.core.settings import BusinessRules
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.mapping import (
    CANONICAL_ORDER_STATUSES,
    MappingError,
    SchemaMapping,
    load_mapping,
)
from support_agent.mcp_db.service import BusinessService
from tests.conftest import ALICE, DEMO, ROOT

DEMO = DEMO


def sqlite_dict() -> dict[str, Any]:
    return yaml.safe_load(
        (DEMO / "config" / "schema_mapping.sqlite.yaml").read_text(encoding="utf-8")
    )


def mapping_without(*entities: str) -> SchemaMapping:
    d = copy.deepcopy(sqlite_dict())
    for name in entities:
        del d["entities"][name]
    return SchemaMapping.model_validate(d)


# --- the shipped template ----------------------------------------------------------------------


def test_the_template_mapping_is_valid_and_uses_only_canonical_statuses():
    m = load_mapping(ROOT / "config" / "schema_mapping.yaml")
    assert {"customer", "order", "order_item"} <= set(m.entities)
    assert set(m.entity("order").status_map.values()) <= CANONICAL_ORDER_STATUSES


def test_a_three_entity_mapping_is_enough():
    m = mapping_without("product", "inventory", "shipment", "return_request")
    assert set(m.entities) == {"customer", "order", "order_item"}


def test_the_required_entities_are_still_required(tmp_path: Path):
    d = sqlite_dict()
    del d["entities"]["order_item"]
    path = tmp_path / "m.yaml"
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    with pytest.raises(MappingError, match="missing required entity 'order_item'"):
        load_mapping(path)


def test_more_order_statuses_and_money_parts_are_accepted():
    d = sqlite_dict()
    order = d["entities"]["order"]
    order["status_map"].update({"WAIT_PAY": "pending_payment", "PART": "partially_shipped"})
    order["status_map"].update({"RET": "returned", "REF": "refunded", "HOLD": "on_hold"})
    order["fields"].update(
        {"subtotal": "grand_total", "shipping_fee": "0", "discount": "0", "tax": "0"}
    )
    SchemaMapping.model_validate(d)


# --- tools degrade, they do not crash ---------------------------------------------------------------


@pytest.fixture
def lean_service(sqlite_url: str, app_config: Any) -> BusinessService:
    adapter = SqlAdapter.from_url(
        sqlite_url,
        mapping_without("product", "inventory", "shipment", "return_request"),
        timezone="Asia/Ho_Chi_Minh",
    )
    return BusinessService(adapter, app_config.business_rules)


async def test_an_order_still_reads_without_the_optional_entities(lean_service: BusinessService):
    got = await lean_service.get_order(ALICE, "1234")
    assert got.ok and got.data["order"]["id"] == "1234" and got.data["items"]
    assert (await lean_service.list_orders(ALICE)).ok


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.get_shipment_status(ALICE, "1234"),
        lambda s: s.search_products(ALICE, "phone"),
        lambda s: s.check_stock(ALICE, sku="PHN-X100"),
        lambda s: s.compare_products(ALICE, ["PHN-X100", "LAP-PRO14"]),
        lambda s: s.prepare_order_draft(ALICE, [], "12 Le Loi, Hanoi", "cod"),
    ],
)
async def test_tools_that_need_missing_data_say_not_supported(lean_service: BusinessService, call):
    result = await call(lean_service)
    assert not result.ok and result.error.code == "NOT_SUPPORTED"


async def test_return_rules_work_without_a_catalogue_or_return_table(
    lean_service: BusinessService,
):
    result = await lean_service.check_return_eligibility(ALICE, "1234")
    assert result.ok and result.data["eligible"] is True


async def test_comparison_skips_availability_when_there_is_no_stock_table(
    sqlite_url: str, app_config: Any
):
    adapter = SqlAdapter.from_url(sqlite_url, mapping_without("inventory"), timezone="UTC")
    try:
        service = BusinessService(adapter, app_config.business_rules)
        result = await service.compare_products(ALICE, ["PHN-X100", "LAP-PRO14"])
        assert result.ok
        assert "availability" not in {r["attribute"] for r in result.data["rows"]}
    finally:
        await adapter.close()


# --- currency ---------------------------------------------------------------------------------------------


def test_one_currency_setting_feeds_refunds_and_orders():
    rules = BusinessRules.model_validate({"currency": "USD"})
    assert rules.refund.currency == "USD" and rules.order.currency == "USD"


def test_a_section_can_still_override_the_shop_currency():
    rules = BusinessRules.model_validate({"currency": "USD", "refund": {"currency": "EUR"}})
    assert rules.refund.currency == "EUR" and rules.order.currency == "USD"


def test_defaults_stay_vnd_and_do_not_leak_between_instances():
    assert BusinessRules().refund.currency == "VND"
    BusinessRules.model_validate({"currency": "USD"})
    assert BusinessRules().refund.currency == "VND"


def test_amounts_may_have_cents():
    rules = BusinessRules.model_validate(
        {"currency": "USD", "refund": {"auto_review_max_amount": 49.99}}
    )
    assert rules.refund.auto_review_max_amount == 49.99


@pytest.mark.parametrize(
    ("amount", "currency", "lang", "expected"),
    [
        (500_000, "VND", "en", "500,000 VND"),
        (500_000, "VND", "vi", "500.000đ"),
        (49.5, "USD", "en", "49.50 USD"),
        (1234.5, "EUR", "vi", "1.234,50 EUR"),
    ],
)
def test_money_is_formatted_for_the_reader(amount, currency, lang, expected):
    assert format_money(amount, currency, lang) == expected


# --- skills carry no numbers of their own ----------------------------------------------------------------


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_every_skill_resolves_all_of_its_placeholders(lang: str):
    facts = skill_facts(BusinessRules(), lang)
    for name in skill_names():
        skill = load_skill(name, lang, facts)
        assert skill is not None and "{{" not in skill.body, name


def test_skill_figures_follow_the_configuration():
    rules = BusinessRules.model_validate(
        {
            "currency": "USD",
            "refund": {"auto_review_max_amount": 100},
            "order": {
                "cod_max_total": 250,
                "max_lines": 3,
                "max_quantity_per_line": 4,
                "payment_methods": ["card", "paypal"],
            },
        }
    )
    order = load_skill("place-order", "en", skill_facts(rules, "en")).body
    assert "250 USD" in order and "3 different products" in order and "card or PayPal" in order
    refund = load_skill("request-refund", "en", skill_facts(rules, "en")).body
    assert "100 USD" in refund and "500,000" not in refund


def test_a_shop_without_a_cash_on_delivery_cap_gets_neutral_wording():
    rules = BusinessRules.model_validate({"order": {"cod_max_total": None}})
    assert "the shop's limit" in skill_facts(rules)["cod_cap"]


def test_unknown_placeholders_are_left_visible_not_silently_dropped():
    assert render("a {{known}} b {{other}}", {"known": "1"}) == "a 1 b {{other}}"


def test_skills_hold_no_hard_coded_shop_figures():
    root = Path(__file__).resolve().parent.parent / "src" / "support_agent" / "agent" / "skills"
    for path in root.glob("*/SKILL.*.md"):
        text = path.read_text(encoding="utf-8")
        for figure in ("500,000", "500.000", "5,000,000", "5.000.000", "VND"):
            assert figure not in text, f"{path.name}: {figure}"
