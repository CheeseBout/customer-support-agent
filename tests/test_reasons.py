"""A reason the customer already gave must reach the first eligibility check, in either language.

Shops give a faulty item a longer period than a change of mind. The model does not reliably pass
the reason on by itself (it did in English and not in Vietnamese in the evaluation), so our own
code spots it and says so in a hint, and a "no" that a reason could change says it is not final.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from support_agent.agent.graph import _hints
from support_agent.agent.prompts import system_prompt
from support_agent.core.capabilities import Offer
from support_agent.core.reasons import fold_words, reasons_in
from support_agent.core.settings import ExchangeRule, ReturnRule, RouterConfig
from support_agent.rag.entities import extract_entities
from support_agent.rules.eligibility import NOT_FINAL_NOTE, check_return_eligibility
from support_agent.rules.exchange import check_exchange_eligibility

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime.fromisoformat("2026-10-20T12:00:00+07:00")


# --- spotting the reason -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thanh RAM bị lỗi, mình muốn hoàn tiền", ["defective"]),
        ("Vòng đeo tay thông minh bị lỗi, tôi trả được không?", ["defective"]),
        ("Laptop bị hỏng màn hình", ["defective"]),
        ("Máy không hoạt động nữa", ["defective"]),
        ("The earbuds stopped working after two days", ["defective"]),
        ("my left bud is dead", ["defective"]),
        ("The screen is faulty", ["defective"]),
        ("Giao sai hàng rồi", ["wrong_item"]),
        ("I received the wrong item", ["wrong_item"]),
        ("Hàng không đúng mô tả", ["not_as_described"]),
        ("Hộp móp méo, hư hỏng khi vận chuyển", ["damaged_in_transit", "defective"]),
        ("It arrived damaged", ["damaged_in_transit"]),
        ("Áo không vừa, nhỏ quá", ["changed_mind"]),
        ("I changed my mind", ["changed_mind"]),
        ("Giày bị lỗi và mình cũng không thích màu", ["defective", "changed_mind"]),
    ],
)
def test_the_reason_in_a_message_is_recognised(text: str, expected: list[str]):
    assert reasons_in(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Xin lỗi, mình muốn hỏi về đơn hàng",  # "sorry" is not a fault
        "Mình muốn đổi sang đầm hồng size M",  # the colour pink is not "broken"
        "Where is my order?",
        "Tôi muốn trả hàng",  # wants to return, says nothing about why
        "",
    ],
)
def test_words_that_only_look_like_a_reason_are_ignored(text: str):
    assert reasons_in(text) == []


def test_a_specific_fault_wins_over_a_plain_one():
    assert reasons_in("hư hỏng khi vận chuyển")[0] == "damaged_in_transit"


def test_a_shop_can_add_its_own_phrases():
    assert reasons_in("Máy bị chập nguồn") == []
    assert reasons_in("Máy bị chập nguồn", {"defective": ["bị chập"]}) == ["defective"]
    assert fold_words("Đã HỎNG") == "da hong"


# --- the hint ------------------------------------------------------------------------------------------


def state(question: str, orders: tuple[str, ...] = ("AB-1",)) -> Any:
    return {"order_ids": list(orders), "question": question}


def reason_hint(hints: list[str]) -> str | None:
    return next((h for h in hints if "already given a reason" in h), None)


def test_a_vietnamese_fault_is_passed_to_the_return_check():
    hint = reason_hint(_hints(state("Vòng đeo tay ở đơn AB-1 bị lỗi, tôi trả được không?")))
    assert hint is not None and 'reason="defective"' in hint and "check_return_eligibility" in hint


def test_an_english_fault_is_passed_to_the_return_check():
    hint = reason_hint(_hints(state("The RAM in order AB-1 is faulty. I'd like a refund.")))
    assert hint is not None and 'reason="defective"' in hint


def test_an_exchange_gets_the_exchange_check_and_the_reason():
    hint = reason_hint(_hints(state("Áo ở đơn AB-1 không vừa, mình muốn đổi size"), exchanges=True))
    assert hint is not None and 'reason="changed_mind"' in hint
    assert "check_exchange_eligibility" in hint and "check_return_eligibility" not in hint


def test_no_hint_without_an_order_or_without_a_reason_or_without_such_a_question():
    assert reason_hint(_hints(state("Máy bị lỗi, trả được không?", orders=()))) is None
    assert reason_hint(_hints(state("Tôi muốn trả hàng đơn AB-1"))) is None  # no reason given
    assert (
        reason_hint(_hints(state("Đơn AB-1 bị lỗi giao hàng ở đâu rồi?"))) is None
    )  # not a return


def test_the_shops_own_phrases_reach_the_hint():
    question = "Máy ở đơn AB-1 bị chập, trả được không?"
    assert reason_hint(_hints(state(question))) is None
    words = RouterConfig(reason_words={"defective": ["bị chập"]}).reason_words
    assert reason_hint(_hints(state(question), reason_words=words)) is not None


# --- a "no" a reason could change is not final ------------------------------------------------------------


def order(delivered: str = "2026-10-10T10:00:00+07:00") -> dict[str, Any]:
    return {
        "id": "A1",
        "customer_id": "c1",
        "status": "delivered",
        "created_at": "2026-10-08T09:00:00+07:00",
        "delivered_at": delivered,
    }


ITEMS = [{"sku": "TEE", "product_name": "Tee", "quantity": 1, "unit_price": 100_000}]


def returns(rule: ReturnRule, reason: str | None = None):
    return check_return_eligibility(
        order=order(),
        items=ITEMS,
        products={"TEE": {"category": "tops"}},
        existing_requests=[],
        requester_id="c1",
        rule=rule,
        now=NOW,
        tz=TZ,
        reason=reason,
    )


LONGER_FOR_A_FAULT = [{"reasons": ["defective"], "days": 30}]


def test_a_no_that_a_fault_could_change_says_it_is_not_final():
    rule = ReturnRule(window_days=7, windows=LONGER_FOR_A_FAULT)  # type: ignore[arg-type]
    plain = returns(rule)
    assert not plain.eligible and plain.note == NOT_FINAL_NOTE
    assert "call this tool again with `reason`" in NOT_FINAL_NOTE


def test_a_no_that_nothing_can_change_has_no_note():
    assert returns(ReturnRule(window_days=7)).note is None
    already = returns(ReturnRule(window_days=7, windows=LONGER_FOR_A_FAULT), "defective")  # type: ignore[arg-type]
    assert already.eligible and already.note is None  # a yes needs no note


def test_the_note_does_not_appear_once_the_longer_period_has_ended_too():
    old = check_return_eligibility(
        order=order("2026-08-01T10:00:00+07:00"),
        items=ITEMS,
        products={"TEE": {"category": "tops"}},
        existing_requests=[],
        requester_id="c1",
        rule=ReturnRule(window_days=7, windows=LONGER_FOR_A_FAULT),  # type: ignore[arg-type]
        now=NOW,
        tz=TZ,
    )
    assert not old.eligible and old.note is None  # not even a fault would help


def test_an_exchange_refused_on_the_plain_period_carries_the_note_too():
    result = check_exchange_eligibility(
        order=order(),
        items=ITEMS,
        products={"TEE": {"category": "tops"}},
        existing_exchanges=[],
        existing_returns=[],
        requester_id="c1",
        rule=ExchangeRule(window_days=7, windows=LONGER_FOR_A_FAULT),  # type: ignore[arg-type]
        now=NOW,
        tz=TZ,
    )
    assert not result.eligible and result.note == NOT_FINAL_NOTE


# --- order ids with several parts -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "ids"),
    [
        ("Đơn AB-20261006-0004 đang ở đâu?", ["AB-20261006-0004"]),
        ("Where is order XY-261008-0004?", ["XY-261008-0004"]),
        ("đơn hàng 1234 và order 2026-0005", ["1234", "2026-0005"]),
        ("order #1234", ["1234"]),
        ("Where is order 1236?", ["1236"]),
    ],
)
def test_an_order_id_is_not_cut_short_at_its_second_hyphen(text: str, ids: list[str]):
    assert extract_entities(text).order_ids == ids


def test_a_shorter_match_of_the_same_id_is_dropped():
    cfg = RouterConfig(order_id_patterns=[r"\b(AB-\d{8}-\d{4})\b", r"\b(AB-\d{8})\b"])
    assert extract_entities("see AB-20261006-0004", cfg).order_ids == ["AB-20261006-0004"]
    # two different orders are both kept
    both = extract_entities("AB-20261006-0004 and AB-20261007-0001", cfg).order_ids
    assert both == ["AB-20261006-0004", "AB-20261007-0001"]


# --- what the model is told ------------------------------------------------------------------------------


def test_the_prompt_tells_the_model_not_to_invent_what_the_shop_lacks():
    text = system_prompt("en", offer=Offer())
    assert "Never say that the shop does not offer" in text and "[NO_INFO]" in text
    assert "`note` saying it is not final" in text
