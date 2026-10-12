"""Exchanging an item (another size, colour or unit) is its own kind of request.

It has its own window, separate from the return window; a limit per order; a check of the item the
customer wants instead; and it must never be judged by the return rules. Covers the rules, the
tool, the request the agent prepares, the skill and the prompt.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

from support_agent.agent.drafting import (
    EDITABLE,
    DraftProposer,
    ProposalError,
    ProposeDraftArgs,
)
from support_agent.agent.graph import _hints
from support_agent.agent.prompts import PROMPT_VERSION, system_prompt
from support_agent.agent.skills import load_skill, skill_catalog
from support_agent.agent.tools import allowed_tools, draft_view
from support_agent.core.capabilities import (
    REQUEST_TYPES,
    Offer,
    available_tools,
    effective_request_types,
)
from support_agent.core.settings import AppConfig, BusinessRules, ExchangeRule
from support_agent.drafts.models import PAYLOAD_MODELS
from support_agent.drafts.repository import SqlDraftRepository
from support_agent.drafts.service import DraftService
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.server import build_server
from support_agent.mcp_db.service import BusinessService
from support_agent.rag.personal import wants_exchange_check, wants_return_check
from support_agent.rules.eligibility import check_return_eligibility
from support_agent.rules.exchange import (
    check_exchange_eligibility,
    check_replacement,
    exchanges_used,
    with_pending_exchanges,
)
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, SECRET
from tests.test_agent import collect, done
from tests.test_agent import fast_config as fast_config  # noqa: F401  (fixtures)
from tests.test_agent import retriever as retriever  # noqa: F401
from tests.test_agent_actions import llm_for, waiting

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
NOW = datetime.fromisoformat("2026-10-20T12:00:00+07:00")


def order(delivered: str = "2026-10-10T10:00:00+07:00", **over: Any) -> dict[str, Any]:
    base = {
        "id": "A1",
        "customer_id": "c1",
        "status": "delivered",
        "created_at": "2026-10-08T09:00:00+07:00",
        "delivered_at": delivered,
    }
    return base | over


ITEMS = [
    {"sku": "TEE-M", "product_name": "Tee", "quantity": 1, "unit_price": 300_000},
    {"sku": "BRA-M", "product_name": "Bra", "quantity": 1, "unit_price": 150_000},
]
PRODUCTS = {
    "TEE-M": {"category": "tops", "group_id": "tee", "active": True},
    "BRA-M": {"category": "underwear", "group_id": "bra", "active": True},
    "TEE-L": {"category": "tops", "group_id": "tee", "active": True, "name": "Tee L"},
    "SHOE-L": {"category": "shoes", "group_id": "shoe", "active": True},
}


def check(rule: ExchangeRule | None = None, **over: Any):
    args: dict[str, Any] = dict(
        order=order(),
        items=ITEMS,
        products=PRODUCTS,
        existing_exchanges=[],
        existing_returns=[],
        requester_id="c1",
        rule=rule or ExchangeRule(excluded_categories=["underwear"]),
        now=NOW,
        tz=TZ,
    )
    return check_exchange_eligibility(**(args | over))


# --- the rules ---------------------------------------------------------------------------------------


def test_an_exchange_has_its_own_window_not_the_return_window():
    """Delivered 10 days ago: past a 7-day return window, inside a 15-day exchange window."""
    returned = check_return_eligibility(
        order=order(),
        items=ITEMS,
        products=PRODUCTS,
        existing_requests=[],
        requester_id="c1",
        rule=ExchangeRule(window_days=7),
        now=NOW,
        tz=TZ,
    )
    assert not returned.eligible and returned.reasons == ["WINDOW_EXPIRED"]  # ended 17 Oct
    exchange = check(ExchangeRule(window_days=15))
    assert exchange.eligible and exchange.reasons == []
    assert exchange.deadline == "2026-10-25T00:00:00+07:00"
    assert exchange.days_remaining == 5  # 4.5 days left, rounded up


def test_the_exchange_window_ends_at_its_own_deadline():
    r = check(ExchangeRule(window_days=7))
    assert not r.eligible and r.reasons == ["WINDOW_EXPIRED"]
    assert r.eligible_items == []


def test_a_fault_can_have_a_longer_exchange_window():
    rule = ExchangeRule(
        window_days=7,
        windows=[{"reasons": ["defective"], "days": 30}],  # type: ignore[list-item]
    )
    plain = check(rule)
    assert not plain.eligible
    assert plain.reason_windows and plain.reason_windows[0]["still_open"]  # a reason would help
    faulty = check(rule, reason="defective")
    assert faulty.eligible and faulty.deadline == "2026-11-09T00:00:00+07:00"


def test_an_order_that_is_not_delivered_cannot_be_exchanged():
    r = check(order=order(status="shipping"))
    assert not r.eligible and "STATUS_NOT_ALLOWED" in r.reasons


def test_an_excluded_category_cannot_be_exchanged():
    r = check(sku="BRA-M")
    assert not r.eligible and "CATEGORY_EXCLUDED" in r.reasons


def test_the_other_items_of_the_order_stay_eligible():
    r = check()
    assert r.eligible and [i["sku"] for i in r.eligible_items] == ["TEE-M"]
    assert "unit_price" not in r.eligible_items[0]  # no money changes hands in an exchange


def test_an_item_with_an_open_return_cannot_also_be_exchanged():
    r = check(existing_returns=[{"status": "pending", "sku": "TEE-M"}])
    assert not r.eligible and "ALREADY_REQUESTED" in r.reasons


def test_another_customers_order_is_not_exchangeable():
    r = check(requester_id="someone-else")
    assert not r.eligible and "NOT_OWNER" in r.reasons


def test_the_limit_per_order_counts_exchanges_not_rows():
    rows = [
        {"id": "EX1", "status": "completed", "sku": "TEE-M"},
        {"id": "EX1", "status": "completed", "sku": "BRA-M"},  # the same exchange, two items
        {"id": "EX2", "status": "rejected", "sku": "TEE-M"},  # turned down: does not count
        {"id": "EX3", "status": "cancelled", "sku": "TEE-M"},
    ]
    assert exchanges_used(rows) == 1
    blocked = check(existing_exchanges=rows)  # default allowance: one per order
    assert not blocked.eligible and blocked.reasons == ["EXCHANGE_LIMIT_REACHED"]
    assert blocked.exchanges_used == 1 and blocked.exchanges_allowed == 1
    assert check(ExchangeRule(max_per_order=2), existing_exchanges=rows).eligible
    assert check(ExchangeRule(max_per_order=None), existing_exchanges=rows).eligible


def test_exchanges_taken_through_the_chat_count_against_the_limit():
    base = check().model_dump(mode="json")
    assert base["eligible"]
    after = with_pending_exchanges(base, 1)
    assert not after["eligible"] and after["reasons"] == ["EXCHANGE_LIMIT_REACHED"]
    assert after["eligible_items"] == [] and after["exchanges_used"] == 1
    assert with_pending_exchanges(base, 0) is base  # nothing pending: untouched
    roomy = check(ExchangeRule(max_per_order=3)).model_dump(mode="json")
    assert with_pending_exchanges(roomy, 1)["eligible"]


# --- the item the customer wants instead -------------------------------------------------------------


def replacement(new_sku: str, *, reason: str | None = None, stock: Any = None, **rule: Any):
    return check_replacement(
        sku="TEE-M",
        new_sku=new_sku,
        products=PRODUCTS,
        stock={"TEE-L": 5} if stock is None else stock,
        rule=ExchangeRule(**rule),
        reason=reason,
    )


def test_another_size_of_the_same_product_can_be_had():
    r = replacement("TEE-L")
    assert r["valid"] and r["problems"] == [] and r["available"] is True and r["name"] == "Tee L"


def test_a_different_product_is_refused_when_only_variants_are_allowed():
    r = replacement("SHOE-L", stock={"SHOE-L": 3})
    assert not r["valid"] and r["problems"] == ["REPLACEMENT_DIFFERENT_PRODUCT"]
    assert replacement("SHOE-L", stock={"SHOE-L": 3}, same_product_only=False)["valid"]


def test_an_item_out_of_stock_cannot_be_had():
    r = replacement("TEE-L", stock={"TEE-L": 0})
    assert not r["valid"] and r["problems"] == ["REPLACEMENT_OUT_OF_STOCK"]
    assert r["available"] is False


def test_unknown_stock_is_not_a_refusal():
    r = check_replacement(
        sku="TEE-M",
        new_sku="TEE-L",
        products=PRODUCTS,
        stock=None,  # the shop keeps no stock data
        rule=ExchangeRule(),
        reason=None,
    )
    assert r["valid"] and r["available"] is None


def test_an_item_that_does_not_exist_cannot_be_had():
    r = replacement("NOPE")
    assert not r["valid"] and r["problems"] == ["REPLACEMENT_NOT_FOUND"]


def test_swapping_an_item_for_itself_needs_a_fault():
    assert replacement("TEE-M", stock={"TEE-M": 2}, reason="changed_mind")["problems"] == [
        "REPLACEMENT_SAME_ITEM"
    ]
    # PRODUCTS holds TEE-M too, so a new unit of a faulty one is fine
    assert replacement("TEE-M", stock={"TEE-M": 2}, reason="defective")["valid"]


# --- configuration ---------------------------------------------------------------------------------------


def test_a_shop_takes_no_exchanges_unless_it_sets_the_rules():
    assert BusinessRules().exchange is None
    rule = ExchangeRule()
    assert rule.window_days == 15 and rule.max_per_order == 1 and rule.same_product_only


def test_exchange_is_a_request_type_and_a_tool_only_for_shops_that_opt_in():
    assert "exchange" in REQUEST_TYPES
    entities = {"customer", "order", "order_item"}
    assert "check_exchange_eligibility" not in available_tools(entities)
    assert "check_exchange_eligibility" in available_tools(entities, exchange=True)
    # a shop that disabled the tool by name does not get it either
    assert "check_exchange_eligibility" not in available_tools(
        entities, ["check_exchange_eligibility"], exchange=True
    )
    # the request type needs the tool, like an order needs the pricing tool
    configured = list(REQUEST_TYPES)
    tools = available_tools(entities)
    assert "exchange" not in effective_request_types(configured, tools)
    assert "exchange" in effective_request_types(
        configured, available_tools(entities, exchange=True)
    )
    assert "exchange" in effective_request_types(configured, None)


def test_the_model_sees_the_exchange_tool_only_when_the_shop_has_it():
    with_it = frozenset({"get_order", "check_exchange_eligibility"})
    assert "check_exchange_eligibility" in allowed_tools("personal", domain=with_it)
    assert "check_exchange_eligibility" not in allowed_tools(
        "personal", domain=frozenset({"get_order"})
    )


# --- the tool over the demo shop -----------------------------------------------------------------------


@pytest.fixture
def rules(app_config: AppConfig) -> BusinessRules:
    return app_config.business_rules.model_copy(
        update={"exchange": ExchangeRule(window_days=30, same_product_only=False)}
    )


@pytest.fixture
def xservice(adapter: SqlAdapter, rules: BusinessRules) -> BusinessService:
    return BusinessService(adapter, rules)


async def test_a_shop_without_exchange_rules_answers_not_supported(service: BusinessService):
    r = await service.check_exchange_eligibility(ALICE, "1234", "EAR-BT20")
    assert not r.ok and r.error is not None and r.error.code == "NOT_SUPPORTED"
    assert "person" in r.error.message  # tells the model what to do instead


async def test_the_exchange_window_is_not_the_return_window(
    service: BusinessService, xservice: BusinessService
):
    """Order 1235 was delivered 20 days ago: past the 7-day return window, inside a 30-day
    exchange window."""
    returned = await service.check_return_eligibility(ALICE, "1235")
    assert returned.ok and returned.data["eligible"] is False
    swapped = await xservice.check_exchange_eligibility(ALICE, "1235")
    assert swapped.ok and swapped.data["eligible"] is True
    assert swapped.data["exchanges_allowed"] == 1 and "refundable_amount" not in swapped.data


async def test_someone_elses_order_looks_like_a_missing_one(xservice: BusinessService):
    mine = await xservice.check_exchange_eligibility(ALICE, "2001")  # Bob's order
    missing = await xservice.check_exchange_eligibility(ALICE, "999999")
    assert not mine.ok and mine.error is not None
    assert mine.error.code == missing.error.code == "NOT_FOUND"  # type: ignore[union-attr]
    assert BOB.user_id != ALICE.user_id


async def test_an_open_return_request_blocks_the_exchange_of_that_item(xservice: BusinessService):
    r = await xservice.check_exchange_eligibility(ALICE, "1240")  # has a pending return
    assert r.ok and r.data["eligible"] is False and "ALREADY_REQUESTED" in r.data["reasons"]


async def test_the_replacement_is_checked_against_stock(xservice: BusinessService):
    ok = await xservice.check_exchange_eligibility(ALICE, "1234", "EAR-BT20", "MOU-M10")
    assert ok.ok and ok.data["replacement"]["valid"] and ok.data["replacement"]["available"]
    gone = await xservice.check_exchange_eligibility(ALICE, "1234", "EAR-BT20", "CHG-65W")
    assert gone.data["replacement"]["problems"] == ["REPLACEMENT_OUT_OF_STOCK"]
    nothing = await xservice.check_exchange_eligibility(ALICE, "1234", "EAR-BT20", "NOPE")
    assert nothing.data["replacement"]["problems"] == ["REPLACEMENT_NOT_FOUND"]
    no_stock_number = ok.data["replacement"]
    assert "quantity" not in no_stock_number  # stock levels stay private


async def test_the_server_lists_the_exchange_tool_only_with_exchange_rules(
    xservice: BusinessService, service: BusinessService
):
    names: dict[str, frozenset[str]] = {}

    async def listing(label: str, svc: BusinessService) -> None:
        async with DomainToolClient(build_server(svc, SECRET), SECRET) as client:
            names[label] = await client.tool_names()

    await listing("with", xservice)
    await listing("without", service)
    assert "check_exchange_eligibility" in names["with"]
    assert "check_exchange_eligibility" not in names["without"]


# --- the request the agent prepares ---------------------------------------------------------------------


@pytest_asyncio.fixture
async def xclient(xservice: BusinessService) -> AsyncIterator[DomainToolClient]:
    ready, stop = asyncio.Event(), asyncio.Event()
    box: dict[str, DomainToolClient] = {}

    async def owner() -> None:
        async with DomainToolClient(build_server(xservice, SECRET), SECRET) as client:
            box["client"] = client
            ready.set()
            await stop.wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), timeout=10)
    yield box["client"]
    stop.set()
    await task


@pytest_asyncio.fixture
async def drafts(rules: BusinessRules, tmp_path: Path) -> AsyncIterator[DraftService]:
    repo = SqlDraftRepository.from_url(f"sqlite+aiosqlite:///{(tmp_path / 'd.db').as_posix()}")
    await repo.create_schema()
    yield DraftService(repo, rules)
    await repo.close()


@pytest.fixture
def proposer(xclient: DomainToolClient, drafts: DraftService, rules: BusinessRules):
    return DraftProposer(xclient, drafts, rules)


def exchange(**kw: Any) -> ProposeDraftArgs:
    return ProposeDraftArgs(draft_type="exchange", **kw)


async def test_an_exchange_is_proposed_with_what_the_customer_gets_instead(proposer: DraftProposer):
    p = await proposer.propose(
        exchange(
            order_id="1234",
            sku="EAR-BT20",
            new_sku="MOU-M10",
            reason_code="changed_mind",
            reason_text="does not suit me",
        ),
        ALICE,
    )
    assert p.draft_type == "exchange" and not p.priority_review
    assert p.payload["order_id"] == "1234" and p.payload["sku"] == "EAR-BT20"
    assert p.payload["new_sku"] == "MOU-M10" and p.payload["reason_code"] == "changed_mind"
    assert p.summary["item"]["sku"] == "EAR-BT20" and p.summary["instead"]["sku"] == "MOU-M10"
    assert p.summary["deadline"] and "amount" not in p.summary  # no money in an exchange
    PAYLOAD_MODELS["exchange"].model_validate(p.payload)  # the stored form is valid


async def test_an_exchange_needs_to_say_what_is_wanted_instead(proposer: DraftProposer):
    with pytest.raises(ProposalError) as err:
        await proposer.propose(exchange(order_id="1234", sku="EAR-BT20"), ALICE)
    assert err.value.code == "INVALID_ARGUMENT"
    with_note = await proposer.propose(
        exchange(order_id="1234", sku="EAR-BT20", reason_text="a bigger size please"), ALICE
    )
    assert with_note.payload["new_sku"] is None  # staff will sort out the detail


async def test_an_exchange_the_rules_refuse_is_not_proposed(proposer: DraftProposer):
    for order_id, sku, code in (
        ("1240", "SPK-BM5", "ALREADY_REQUESTED"),  # a return is already open
        ("1236", "LAP-PRO14", "STATUS_NOT_ALLOWED"),  # still on its way
    ):
        with pytest.raises(ProposalError) as err:
            await proposer.propose(
                exchange(order_id=order_id, sku=sku, new_sku="CASE-X100", reason_text="a swap"),
                ALICE,
            )
        assert err.value.code == "NOT_ELIGIBLE" and code in err.value.details["reasons"], order_id


async def test_the_window_closes_when_the_exchange_period_ends(
    adapter: SqlAdapter, rules: BusinessRules
):
    short = rules.model_copy(update={"exchange": ExchangeRule(window_days=3)})
    r = await BusinessService(adapter, short).check_exchange_eligibility(ALICE, "1235")
    assert r.ok and r.data["eligible"] is False and "WINDOW_EXPIRED" in r.data["reasons"]


async def test_an_item_that_cannot_be_had_instead_is_refused(proposer: DraftProposer):
    with pytest.raises(ProposalError) as gone:
        await proposer.propose(
            exchange(order_id="1234", sku="EAR-BT20", new_sku="CHG-65W", reason_text="x y z"),
            ALICE,
        )
    assert gone.value.code == "OUT_OF_STOCK"
    with pytest.raises(ProposalError) as unknown:
        await proposer.propose(
            exchange(order_id="1234", sku="EAR-BT20", new_sku="NOPE", reason_text="x y z"), ALICE
        )
    assert unknown.value.code == "NOT_ELIGIBLE"


async def test_a_second_exchange_on_the_same_order_is_refused(
    proposer: DraftProposer, drafts: DraftService
):
    args = exchange(order_id="1234", sku="EAR-BT20", new_sku="MOU-M10", reason_text="a swap")
    first = await proposer.propose(args, ALICE)
    draft, created = await drafts.create(ALICE, "exchange", first.payload, "s1")
    assert created and draft.type == "exchange" and draft.sku_set() == {"EAR-BT20"}
    with pytest.raises(ProposalError) as err:
        await proposer.propose(args, ALICE)
    assert err.value.code == "NOT_ELIGIBLE"
    assert "EXCHANGE_LIMIT_REACHED" in err.value.details["reasons"]


async def test_a_withdrawn_exchange_gives_the_allowance_back(
    proposer: DraftProposer, drafts: DraftService
):
    args = exchange(order_id="1234", sku="EAR-BT20", new_sku="MOU-M10", reason_text="a swap")
    first = await proposer.propose(args, ALICE)
    draft, _ = await drafts.create(ALICE, "exchange", first.payload, "s1")
    await drafts.cancel(ALICE, draft.id)
    assert (await proposer.propose(args, ALICE)).draft_type == "exchange"


async def test_the_customer_may_change_the_item_wanted_while_confirming(proposer: DraftProposer):
    assert EDITABLE["exchange"] == {"reason_code", "reason_text", "new_sku"}
    first = await proposer.propose(
        exchange(order_id="1234", sku="EAR-BT20", new_sku="MOU-M10", reason_text="a swap"), ALICE
    )
    again = await proposer.edited(first, {"new_sku": "CASE-X100"}, ALICE)
    assert again.payload["new_sku"] == "CASE-X100"  # every check ran again
    with pytest.raises(ProposalError) as err:
        await proposer.edited(first, {"order_id": "1240"}, ALICE)  # what the checks ran on
    assert err.value.code == "INVALID_ARGUMENT"
    with pytest.raises(ProposalError) as stock:
        await proposer.edited(first, {"new_sku": "CHG-65W"}, ALICE)
    assert stock.value.code == "OUT_OF_STOCK"


async def test_a_shop_that_does_not_allow_exchanges_refuses_the_request(
    xclient: DomainToolClient, drafts: DraftService, rules: BusinessRules
):
    closed = DraftProposer(xclient, drafts, rules, request_types=["refund", "return"])
    with pytest.raises(ProposalError) as err:
        await closed.propose(exchange(order_id="1234", sku="EAR-BT20", reason_text="x y z"), ALICE)
    assert err.value.code == "NOT_SUPPORTED"


async def test_a_stored_exchange_is_shown_to_the_customer_without_money(
    proposer: DraftProposer, drafts: DraftService
):
    p = await proposer.propose(
        exchange(order_id="1234", sku="EAR-BT20", new_sku="MOU-M10", reason_text="a swap"), ALICE
    )
    draft, _ = await drafts.create(ALICE, "exchange", p.payload, "s9")
    view = draft_view(draft)
    assert view["type"] == "exchange" and view["sku"] == "EAR-BT20" and view["new_sku"] == "MOU-M10"
    assert "amount" not in view and "total" not in view


# --- the words the model is given ------------------------------------------------------------------------


def test_an_exchange_question_is_not_taken_for_a_return_question():
    for q in ("Can I exchange the size of order 1234?", "Mình muốn đổi size áo ở đơn 1234"):
        assert wants_exchange_check(q) and not wants_return_check(q)
    for q in ("Can I return order 1234?", "Hoàn tiền đơn 1234 được không", "đổi trả đơn 1234"):
        assert wants_return_check(q)
    assert not wants_exchange_check("Where is order 1234?")


def test_the_hint_for_an_exchange_names_the_exchange_tool_and_only_if_the_shop_has_it():
    state: Any = {"order_ids": ["1234"], "question": "I want to exchange the size of order 1234"}
    on = " ".join(_hints(state, exchanges=True))
    assert (
        "check_exchange_eligibility" in on
        and "check_return_eligibility" not in on.split("Never judge")[0]
    )
    off = " ".join(_hints(state, exchanges=False))
    assert "check_exchange_eligibility" not in off and "check_return_eligibility" not in off


def test_the_prompt_no_longer_sends_exchanges_to_the_return_rules():
    assert PROMPT_VERSION == "v10"
    text = system_prompt("en", offer=Offer())
    assert "check_exchange_eligibility" in text
    assert "returning, refunding or exchanging" not in text
    assert "send the item back or exchange it" not in text
    assert "pass that as `reason` in the FIRST call" in text


def test_the_exchange_skill_is_offered_only_to_shops_that_take_exchanges():
    names = [n for n, _ in skill_catalog(Offer())]
    assert "exchange-item" in names
    without = [n for n, _ in skill_catalog(Offer(request_types=("refund", "return")))]
    assert "exchange-item" not in without and "return-item" in without
    for lang in ("en", "vi"):
        skill = load_skill("exchange-item", lang)
        assert skill is not None and skill.lang == lang
        assert "check_exchange_eligibility" in skill.body and "propose_draft" in skill.body
        assert "new_sku" in skill.body and skill.requires == ("request:exchange",)
        ret = load_skill("return-item", lang)
        assert ret is not None and "exchange-item" in ret.body  # points exchanges elsewhere
        assert "reason" in ret.body


# --- through the agent ---------------------------------------------------------------------------------

EXCHANGE = (
    "propose_draft",
    {
        "draft_type": "exchange",
        "order_id": "1234",
        "sku": "EAR-BT20",
        "new_sku": "MOU-M10",
        "reason_code": "changed_mind",
        "reason_text": "I would rather have the mouse",
    },
)


@pytest_asyncio.fixture
async def chat(
    retriever: Any,
    xclient: DomainToolClient,
    proposer: DraftProposer,
    drafts: DraftService,
    fast_config: AppConfig,
):
    from langgraph.checkpoint.memory import InMemorySaver

    from support_agent.agent.agent import SupportAgent
    from support_agent.agent.graph import AgentDeps, build_graph

    def build(llm: Any) -> SupportAgent:
        deps = AgentDeps(
            model=llm.model,
            retriever=retriever,
            client=xclient,
            config=fast_config,
            drafts=drafts,
            proposer=proposer,
            domain_tools=EXCHANGE_SHOP_TOOLS,
        )
        return SupportAgent(
            graph=build_graph(deps, InMemorySaver()),
            retriever=retriever,
            config=fast_config,
            drafts=drafts,
        )

    return build


EXCHANGE_SHOP_TOOLS = frozenset(
    {"get_order", "list_orders", "check_return_eligibility", "check_exchange_eligibility"}
)


async def test_an_exchange_is_confirmed_by_the_customer_then_stored(chat, drafts: DraftService):
    llm = llm_for({"tools": [EXCHANGE]}, {"text": "Your exchange request was submitted."})
    agent = chat(llm)
    events = await collect(
        agent.stream("I want to swap the earbuds of order 1234", ALICE, session_id="x1")
    )
    stop = waiting(events)
    assert stop["kind"] == "confirm_draft" and stop["draft_type"] == "exchange"
    assert stop["editable_fields"] == ["new_sku", "reason_code", "reason_text"]
    assert stop["summary"]["item"]["sku"] == "EAR-BT20"
    assert stop["summary"]["instead"]["sku"] == "MOU-M10"
    assert await drafts.list_for(ALICE) == []  # asking is not submitting

    result = done(await collect(agent.resume(ALICE, stop["session_id"], stop["id"], "approve")))
    (stored,) = await drafts.list_for(ALICE)
    assert stored.type == "exchange" and stored.status == "pending"
    assert stored.payload["new_sku"] == "MOU-M10"
    assert result["drafts"][0]["type"] == "exchange"


async def test_the_model_is_pointed_at_the_exchange_rules_for_a_swap(chat):
    llm = llm_for({"text": "Let me look."})
    agent = chat(llm)
    await collect(agent.stream("Can I exchange the size of order 1234?", ALICE, session_id="x2"))
    system = str(llm.agent_calls()[0][0].content)
    assert "check_exchange_eligibility" in system and "Never judge an exchange with" in system
    assert system.count("This is about returning or refunding") == 0


# --- exchanges the shop already holds --------------------------------------------------------------


async def _service_with_exchange_rows(
    sqlite_url: str, rules: BusinessRules, rows: list[tuple[str, str, str, str]]
) -> tuple[BusinessService, SqlAdapter]:
    import sqlalchemy as sa
    import yaml
    from sqlalchemy.ext.asyncio import create_async_engine

    from support_agent.mcp_db.mapping import SchemaMapping
    from tests.conftest import DEMO

    engine = create_async_engine(sqlite_url)
    async with engine.begin() as conn:
        await conn.execute(
            sa.text("CREATE TABLE shop_exchanges (id TEXT, order_code TEXT, sku TEXT, state TEXT)")
        )
        for row in rows:
            await conn.execute(
                sa.text("INSERT INTO shop_exchanges VALUES (:i, :o, :s, :t)"),
                dict(zip("iost", row, strict=True)),
            )
    await engine.dispose()
    raw = yaml.safe_load((DEMO / "config" / "schema_mapping.sqlite.yaml").read_text("utf-8"))
    raw["entities"]["exchange_request"] = {
        "table": "shop_exchanges",
        "fields": {"id": "id", "order_id": "order_code", "sku": "sku", "status": "state"},
        "status_map": {"DONE": "completed", "NO": "rejected"},
    }
    adapter = SqlAdapter.from_url(
        sqlite_url, SchemaMapping.model_validate(raw), timezone="Asia/Ho_Chi_Minh"
    )
    return BusinessService(adapter, rules), adapter


async def test_exchanges_the_shop_already_holds_use_up_the_allowance(
    sqlite_url: str, rules: BusinessRules
):
    service, adapter = await _service_with_exchange_rows(
        sqlite_url, rules, [("EX1", "1234", "EAR-BT20", "DONE")]
    )
    try:
        r = await service.check_exchange_eligibility(ALICE, "1234", "EAR-BT20")
        assert r.ok and r.data["eligible"] is False
        assert r.data["reasons"] == ["EXCHANGE_LIMIT_REACHED"]
        assert r.data["exchanges_used"] == 1 and r.data["exchanges_allowed"] == 1
        other = await service.check_exchange_eligibility(ALICE, "1235", "CHG-65W")
        assert other.data["eligible"] is True and other.data["exchanges_used"] == 0
    finally:
        await adapter.close()


async def test_an_exchange_the_shop_turned_down_does_not_count(
    sqlite_url: str, rules: BusinessRules
):
    service, adapter = await _service_with_exchange_rows(
        sqlite_url, rules, [("EX1", "1234", "EAR-BT20", "NO")]
    )
    try:
        r = await service.check_exchange_eligibility(ALICE, "1234", "EAR-BT20")
        assert r.data["eligible"] is True and r.data["exchanges_used"] == 0
    finally:
        await adapter.close()
