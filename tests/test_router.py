from __future__ import annotations

import pytest

from support_agent.core.settings import RouterConfig
from support_agent.rag.entities import Entities, extract_entities
from support_agent.rag.router import heuristic_route, route_query
from tests.fakes import SupportFakeLLM

CFG = RouterConfig()


# --- entity extraction (FR-007) ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text,orders",
    [
        ("Where is order #1234?", ["1234"]),
        ("Đơn hàng #1234 của tôi đang ở đâu?", ["1234"]),
        ("order 5678 hasn't arrived", ["5678"]),
        ("Order ID: 90210", ["90210"]),
        ("đơn hàng số 4455 giao chưa", ["4455"]),
        ("mã đơn: 7788", ["7788"]),
        ("don hang 1234 o dau", ["1234"]),
        ("Đơn 1234 và đơn 5678 đều chưa tới", ["1234", "5678"]),
        ("#1234 and #1234 again", ["1234"]),
        ("my order ORD-9981 is late", ["ORD-9981"]),
        ("ord-12345 please", ["ORD-12345"]),
        ("DH20251 bị lỗi", ["DH20251"]),
        ("How many days do I have to return an item?", []),
        ("I have 3 items in order 12", []),  # too short to be an id
        ("see #help", []),  # a hash without digits is not an order
        ("I paid 350000 yesterday", []),
    ],
)
def test_order_id_extraction(text: str, orders: list[str]):
    assert extract_entities(text).order_ids == orders


@pytest.mark.parametrize(
    "text,skus",
    [
        ("Do you have PHN-X100 in stock?", ["PHN-X100"]),
        ("so sánh LAP-PRO14 và LAP-AIR13", ["LAP-PRO14", "LAP-AIR13"]),
        ("Còn hàng GC-500K không?", ["GC-500K"]),
        ("my order ORD-9981", []),  # an order code is not a SKU
        ("I love my t-shirt and e-mail", []),  # lowercase hyphenated words are not SKUs
        ("PHN-X100 PHN-X100", ["PHN-X100"]),
    ],
)
def test_sku_extraction(text: str, skus: list[str]):
    assert extract_entities(text).skus == skus


def test_custom_patterns_extend_the_builtins():
    cfg = RouterConfig(order_id_patterns=[r"\bINV/(\d{4}/\d{4})\b"], sku_pattern=r"\b(\d{8})\b")
    e = extract_entities("invoice INV/2026/0042 for item 12345678", cfg)
    assert e.order_ids == ["2026/0042"] and e.skus == ["12345678"]


def test_entities_merge_and_empty():
    merged = Entities(["1"], ["A-1"]).merge(Entities(["2", "1"], ["A-1", "B-2"]))
    assert merged.order_ids == ["1", "2"] and merged.skus == ["A-1", "B-2"]
    assert Entities().empty and not merged.empty


# --- heuristic fallback --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,route",
    [
        ("What is your return policy?", "policy"),
        ("Chính sách bảo hành thế nào?", "policy"),
        ("Phí vận chuyển bao nhiêu", "policy"),
        ("Where is my order?", "personal"),
        ("Đơn hàng của tôi đang ở đâu", "personal"),
        ("Is order #1234 still eligible for return?", "combined"),
        ("Hello!", "chitchat"),
        ("Xin chào", "chitchat"),
        ("Who won the world cup?", "out_of_scope"),
    ],
)
def test_heuristic_routing(text: str, route: str):
    assert heuristic_route(text, extract_entities(text)).route == route


# --- LLM router --------------------------------------------------------------------------------------


async def test_llm_decision_is_used_with_its_confidence():
    d = await route_query(
        SupportFakeLLM(route="policy", confidence=0.83).model, "How long is the warranty?", CFG
    )
    assert (d.route, d.confidence, d.source) == ("policy", 0.83, "llm")


async def test_regex_entities_are_always_present():
    d = await route_query(SupportFakeLLM(route="personal").model, "Where is order #1234?", CFG)
    assert d.entities.order_ids == ["1234"]


async def test_order_id_upgrades_policy_to_combined():
    d = await route_query(SupportFakeLLM(route="policy").model, "Can I return order #1234?", CFG)
    assert d.route == "combined"


@pytest.mark.parametrize("llm_route", ["chitchat", "out_of_scope"])
async def test_order_id_overrides_dismissive_routes(llm_route: str):
    d = await route_query(SupportFakeLLM(route=llm_route).model, "thanks, but order #1234?", CFG)
    assert d.route == "personal"


async def test_policy_route_stays_when_no_order_id():
    assert (
        await route_query(SupportFakeLLM(route="policy").model, "return policy?", CFG)
    ).route == "policy"


async def test_hallucinated_ids_from_the_llm_are_dropped():
    from tests.fakes import ScriptedChatModel, ai_json

    model = ScriptedChatModel(
        responder=lambda m, t: ai_json(
            {
                "route": "personal",
                "confidence": 0.9,
                "order_ids": ["#9999", "1234"],
                "skus": ["FAKE-SKU"],
            }
        )
    )
    d = await route_query(model, "where is order 1234", CFG)
    assert d.entities.order_ids == ["1234"] and d.entities.skus == []


async def test_llm_failure_degrades_to_heuristics_instead_of_failing():
    d = await route_query(
        SupportFakeLLM(route_error=RuntimeError("503")).model, "What is the return policy?", CFG
    )
    assert d.source == "heuristic" and d.route == "policy"


async def test_no_model_uses_heuristics():
    d = await route_query(None, "Where is order #1234?", CFG)
    assert d.source == "heuristic" and d.route == "personal" and d.entities.order_ids == ["1234"]
