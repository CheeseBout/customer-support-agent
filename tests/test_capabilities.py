"""The assistant offers only what the shop's data and settings allow; parcels; product variants."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
import yaml
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.agent.agent import SupportAgent
from support_agent.agent.drafting import DraftProposer, ProposalError, ProposeDraftArgs
from support_agent.agent.graph import AgentDeps, build_graph
from support_agent.agent.prompts import system_prompt
from support_agent.agent.skills import skill_catalog
from support_agent.agent.tools import allowed_tools, schema_tools_for
from support_agent.core.capabilities import (
    REQUEST_TYPES,
    TOOL_NEEDS,
    Offer,
    available_tools,
    effective_request_types,
)
from support_agent.core.settings import AppConfig, CapabilitiesConfig
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.mapping import SchemaMapping
from support_agent.mcp_db.server import build_server
from support_agent.mcp_db.service import BusinessService
from support_agent.tools.client import DomainToolClient
from support_agent.tools.langchain_tools import DOMAIN_TOOL_NAMES
from tests.conftest import ALICE, DEMO, SECRET
from tests.fakes import AgentFakeLLM
from tests.test_agent import collect, done
from tests.test_agent import fast_config as fast_config  # noqa: F401  (fixtures)
from tests.test_agent import retriever as retriever  # noqa: F401

CORE = {"customer", "order", "order_item"}
LEAN_TOOLS = {
    "get_order",
    "list_orders",
    "check_return_eligibility",
    "check_warranty_eligibility",
}


def demo_mapping(*drop: str) -> SchemaMapping:
    raw = yaml.safe_load((DEMO / "config" / "schema_mapping.sqlite.yaml").read_text("utf-8"))
    for name in drop:
        del raw["entities"][name]
    return SchemaMapping.model_validate(raw)


# --- which tools exist ------------------------------------------------------------------------


def test_the_tool_table_covers_exactly_the_tools_the_agent_knows():
    assert set(TOOL_NEEDS) == set(DOMAIN_TOOL_NAMES)


def test_a_shop_with_only_the_required_entities_gets_the_order_and_rule_tools():
    assert available_tools(CORE) == LEAN_TOOLS


@pytest.mark.parametrize(
    ("extra", "gained"),
    [
        ({"shipment"}, {"get_shipment_status"}),
        ({"product"}, {"search_products", "compare_products"}),
        ({"inventory"}, set()),  # stock is meaningless without a catalogue
        (
            {"product", "inventory"},
            {"search_products", "compare_products", "check_stock", "prepare_order_draft"},
        ),
    ],
)
def test_each_optional_entity_unlocks_its_tools(extra: set[str], gained: set[str]):
    assert available_tools(CORE | extra) == LEAN_TOOLS | gained


def test_a_shop_can_switch_tools_off_and_a_typo_is_an_error():
    everything = CORE | {"shipment", "product", "inventory"}
    assert "compare_products" not in available_tools(everything, ["compare_products"])
    with pytest.raises(ValueError, match="unknown tools"):
        available_tools(everything, ["compare_product"])
    with pytest.raises(ValueError, match="unknown tools"):
        CapabilitiesConfig(disabled_tools=["nope"])


def test_orders_need_a_priced_catalogue():
    assert effective_request_types(REQUEST_TYPES, None) == REQUEST_TYPES
    lean = frozenset(LEAN_TOOLS)
    assert effective_request_types(REQUEST_TYPES, lean) == (
        "refund",
        "return",
        "warranty",
        "handoff",
    )
    assert effective_request_types(["refund"], None) == ("refund",)
    assert effective_request_types([], None) == ()


def test_the_default_capabilities_change_nothing(app_config: AppConfig):
    assert app_config.capabilities.disabled_tools == []
    assert tuple(app_config.capabilities.request_types) == REQUEST_TYPES


# --- the tool server --------------------------------------------------------------------------------


@pytest_asyncio.fixture
async def lean_client(sqlite_url: str, app_config: AppConfig) -> AsyncIterator[DomainToolClient]:
    adapter = SqlAdapter.from_url(
        sqlite_url,
        demo_mapping("product", "inventory", "shipment", "return_request"),
        timezone="Asia/Ho_Chi_Minh",
    )
    service = BusinessService(adapter, app_config.business_rules)
    ready, stop = asyncio.Event(), asyncio.Event()
    box: dict[str, DomainToolClient] = {}

    async def owner() -> None:
        async with DomainToolClient(build_server(service, SECRET), SECRET) as client:
            box["client"] = client
            ready.set()
            await stop.wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), timeout=10)
    yield box["client"]
    stop.set()
    await task
    await adapter.close()


async def test_the_server_lists_only_the_tools_the_data_supports(lean_client: DomainToolClient):
    assert await lean_client.tool_names() == LEAN_TOOLS


async def test_the_full_demo_server_lists_every_tool(tool_client: DomainToolClient):
    assert await tool_client.tool_names() == set(DOMAIN_TOOL_NAMES)


async def test_a_disabled_tool_is_not_registered(service: BusinessService):
    ready, stop = asyncio.Event(), asyncio.Event()
    seen: dict[str, frozenset[str]] = {}

    async def owner() -> None:
        server = build_server(service, SECRET, disabled=["compare_products"])
        async with DomainToolClient(server, SECRET) as client:
            seen["names"] = await client.tool_names()
            ready.set()
            await stop.wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), timeout=10)
    stop.set()
    await task
    assert seen["names"] == set(DOMAIN_TOOL_NAMES) - {"compare_products"}


# --- what the model is shown ----------------------------------------------------------------------------


def test_the_agent_tool_lists_follow_the_shop_data():
    lean = frozenset(LEAN_TOOLS)
    full = allowed_tools("personal")
    assert allowed_tools("personal", domain=lean) == full - (DOMAIN_TOOL_NAMES - lean)
    names = {t.name for t in schema_tools_for("combined", domain=lean)}
    assert "get_order" in names and "search_policy" in names and "propose_draft" in names
    assert not names & {"get_shipment_status", "check_stock", "compare_products"}
    assert allowed_tools("personal", domain=None) == full  # unknown means everything


def test_skills_the_shop_cannot_follow_are_not_offered():
    everything = {n for n, _ in skill_catalog()}
    assert {"place-order", "compare-products", "request-refund"} <= everything
    lean = Offer(domain_tools=frozenset(LEAN_TOOLS))
    assert {n for n, _ in skill_catalog(lean)} == everything - {"place-order", "compare-products"}
    no_refunds = Offer(request_types=("order", "return", "warranty"))
    assert "request-refund" not in {n for n, _ in skill_catalog(no_refunds)}


def test_the_prompt_says_what_is_missing_only_when_something_is():
    plain = system_prompt("en")
    assert system_prompt("en", offer=Offer()) == plain  # nothing missing: byte for byte the same
    lean = system_prompt("en", offer=Offer(domain_tools=frozenset(LEAN_TOOLS)))
    assert "these tools do not exist here" in lean and "get_shipment_status" in lean
    assert "place-order:" not in lean and "place-order:" in plain
    no_orders = system_prompt("en", offer=Offer(request_types=("refund", "return", "warranty")))
    assert "requests of these kinds cannot be submitted here: order" in no_orders


def test_with_actions_off_the_prompt_does_not_repeat_what_it_already_says():
    text = system_prompt("en", actions=False, offer=Offer(request_types=()))
    assert "cannot be submitted in this deployment" in text
    assert "requests of these kinds" not in text


# --- the agent --------------------------------------------------------------------------------------------


@pytest_asyncio.fixture
async def lean_agent(
    lean_client: DomainToolClient,
    retriever: Any,  # noqa: F811
    fast_config: AppConfig,  # noqa: F811
    tmp_path: Path,
) -> AsyncIterator[Callable[[AgentFakeLLM], SupportAgent]]:
    from support_agent.drafts.repository import SqlDraftRepository
    from support_agent.drafts.service import DraftService

    repo = SqlDraftRepository.from_url(f"sqlite+aiosqlite:///{(tmp_path / 'drafts.db').as_posix()}")
    await repo.create_schema()
    drafts = DraftService(repo, fast_config.business_rules)
    domain = await lean_client.tool_names()
    request_types = effective_request_types(REQUEST_TYPES, domain)
    proposer = DraftProposer(
        lean_client, drafts, fast_config.business_rules, request_types=request_types
    )

    def build(llm: AgentFakeLLM) -> SupportAgent:
        deps = AgentDeps(
            model=llm.model,
            retriever=retriever,
            client=lean_client,
            config=fast_config,
            drafts=drafts,
            proposer=proposer,
            domain_tools=domain,
        )
        return SupportAgent(
            graph=build_graph(deps, InMemorySaver()),
            retriever=retriever,
            config=fast_config,
            drafts=drafts,
        )

    yield build
    await repo.close()


async def test_a_shop_without_a_catalogue_never_shows_catalogue_tools_to_the_model(lean_agent):
    llm = AgentFakeLLM([{"text": "Your order was delivered."}], route="personal")
    events = await collect(lean_agent(llm).stream("Where is order 1234?", ALICE, session_id="s"))
    assert done(events)
    offered = {t for bind in llm.agent_binds() for t in bind["tools"]}
    assert "get_order" in offered and "propose_draft" in offered
    assert not offered & {
        "get_shipment_status",
        "check_stock",
        "search_products",
        "compare_products",
    }
    prompt = str(llm.agent_calls()[0][0].content)
    assert "these tools do not exist here" in prompt
    assert "requests of these kinds cannot be submitted here: order" in prompt
    assert "place-order:" not in prompt


async def test_a_call_to_a_tool_that_does_not_exist_here_is_refused(lean_agent):
    llm = AgentFakeLLM(
        [
            {"tools": [("get_shipment_status", {"order_id": "1236"})]},
            {"text": "I cannot check shipments here."},
        ],
        route="personal",
    )
    events = await collect(lean_agent(llm).stream("Where is 1236?", ALICE, session_id="s"))
    ends = [e.data for e in events if e.kind == "tool_end"]
    assert ends and ends[0]["ok"] is False
    assert done(events)


# --- request types ---------------------------------------------------------------------------------------------------


async def test_a_request_type_the_shop_does_not_take_is_refused_before_any_check(
    tool_client: DomainToolClient, app_config: AppConfig
):
    proposer = DraftProposer(
        tool_client,
        drafts=None,  # type: ignore[arg-type]  # never reached: the type is refused first
        rules=app_config.business_rules,
        request_types=["refund", "return"],
    )
    args = ProposeDraftArgs(draft_type="warranty", order_id="1234", sku="EAR-BT20")
    with pytest.raises(ProposalError) as exc:
        await proposer.propose(args, ALICE)
    assert exc.value.code == "NOT_SUPPORTED" and "warranty" in exc.value.message


async def test_with_no_request_types_the_agent_offers_no_request_tools(
    tool_client: DomainToolClient,
    retriever: Any,  # noqa: F811
    fast_config: AppConfig,  # noqa: F811
    tmp_path: Path,
):
    from support_agent.drafts.repository import SqlDraftRepository
    from support_agent.drafts.service import DraftService

    repo = SqlDraftRepository.from_url(f"sqlite+aiosqlite:///{(tmp_path / 'd.db').as_posix()}")
    await repo.create_schema()
    drafts = DraftService(repo, fast_config.business_rules)
    proposer = DraftProposer(tool_client, drafts, fast_config.business_rules, request_types=[])
    deps = AgentDeps(
        model=(llm := AgentFakeLLM([{"text": "ok"}], route="personal")).model,
        retriever=retriever,
        client=tool_client,
        config=fast_config,
        drafts=drafts,
        proposer=proposer,
    )
    agent = SupportAgent(
        graph=build_graph(deps, InMemorySaver()),
        retriever=retriever,
        config=fast_config,
        drafts=drafts,
    )
    await collect(agent.stream("hello there", ALICE, session_id="s"))
    offered = {t for bind in llm.agent_binds() for t in bind["tools"]}
    assert "propose_draft" not in offered and "get_order" in offered
    assert "cannot be submitted in this deployment" in str(llm.agent_calls()[0][0].content)
    await repo.close()


# --- several parcels --------------------------------------------------------------------------------------------------


async def add_shipments(url: str, rows: list[tuple[str, str, str, str]]) -> None:
    """(order, carrier, tracking, status) with increasing update times."""
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        for i, (order, carrier, tracking, status) in enumerate(rows):
            await conn.execute(
                sa.text(
                    "INSERT INTO shipments (order_code, carrier, tracking_no, status, updated_at) "
                    "VALUES (:o, :c, :t, :s, :u)"
                ),
                {
                    "o": order,
                    "c": carrier,
                    "t": tracking,
                    "s": status,
                    "u": f"2026-10-0{i + 1} 10:00:00",
                },
            )
    await engine.dispose()


async def test_an_order_sent_in_two_parcels_reports_both(sqlite_url: str, service: BusinessService):
    await add_shipments(
        sqlite_url,
        [("1237", "GHN", "GHN-A", "DA_GIAO"), ("1237", "GHTK", "GHTK-B", "DANG_VAN_CHUYEN")],
    )
    data = (await service.get_shipment_status(ALICE, "1237")).data
    assert data["shipment_count"] == 2 and data["all_delivered"] is False
    assert [s["tracking_code"] for s in data["shipments"]] == ["GHTK-B", "GHN-A"]  # newest first
    assert data["shipment"]["tracking_code"] == "GHTK-B"  # the old single-parcel field still works
    assert "2 parcels" in data["note"]


async def test_rows_that_share_a_tracking_code_are_updates_of_one_parcel(
    sqlite_url: str, service: BusinessService
):
    await add_shipments(
        sqlite_url,
        [("1237", "GHN", "GHN-A", "CHO_LAY_HANG"), ("1237", "GHN", "GHN-A", "DANG_VAN_CHUYEN")],
    )
    data = (await service.get_shipment_status(ALICE, "1237")).data
    assert "shipments" not in data and "shipment_count" not in data
    assert data["shipment"]["status"] == "in_transit"  # the latest update


async def test_all_parcels_delivered_is_reported(sqlite_url: str, service: BusinessService):
    await add_shipments(
        sqlite_url, [("1237", "GHN", "A", "DA_GIAO"), ("1237", "GHTK", "B", "DA_GIAO")]
    )
    assert (await service.get_shipment_status(ALICE, "1237")).data["all_delivered"] is True


async def test_a_single_parcel_looks_exactly_as_before(service: BusinessService):
    data = (await service.get_shipment_status(ALICE, "1236")).data
    assert set(data) == {"order_id", "order_status", "shipment"}


async def test_the_adapter_still_offers_the_newest_shipment_row(adapter: SqlAdapter):
    assert (await adapter.get_shipment("1236"))["carrier"] == "GHTK"
    assert await adapter.get_shipment("1237") is None
    assert len(await adapter.get_shipments("1236")) == 1


# --- product variants -------------------------------------------------------------------------------------------


@pytest_asyncio.fixture
async def variant_service(sqlite_url: str, app_config: AppConfig) -> AsyncIterator[BusinessService]:
    """The demo shop plus a T-shirt sold in three sizes, as a shop with variants would store it."""
    engine = create_async_engine(sqlite_url)
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "CREATE TABLE variants (sku TEXT, name TEXT, description TEXT, category_name TEXT, "
                "price INTEGER, specs_json TEXT, is_active INTEGER, family TEXT, opts TEXT)"
            )
        )
        await conn.execute(sa.text("CREATE TABLE variant_stock (sku TEXT, on_hand INTEGER)"))
        shirts = [
            ("TEE-S", "Basic tee", "S", 199_000, 4),
            ("TEE-M", "Basic tee", "M", 199_000, 0),
            ("TEE-XL", "Basic tee", "XL", 219_000, 12),
        ]
        for sku, name, size, price, stock in shirts:
            await conn.execute(
                sa.text(
                    "INSERT INTO variants VALUES (:s, :n, 'Cotton t-shirt', 'fashion', :p, '{}', 1, "
                    "'TEE', :o)"
                ),
                {"s": sku, "n": name, "p": price, "o": f'{{"size": "{size}"}}'},
            )
            await conn.execute(
                sa.text("INSERT INTO variant_stock VALUES (:s, :q)"), {"s": sku, "q": stock}
            )
        await conn.execute(
            sa.text(
                "INSERT INTO variants VALUES ('MUG-1', 'Mug', 'Ceramic mug', 'home', 99000, '{}', 1, "
                "NULL, NULL)"
            )
        )
        await conn.execute(sa.text("INSERT INTO variant_stock VALUES ('MUG-1', 30)"))
    await engine.dispose()

    raw = copy.deepcopy(
        yaml.safe_load((DEMO / "config" / "schema_mapping.sqlite.yaml").read_text("utf-8"))
    )
    raw["entities"]["product"] = {
        "table": "variants",
        "fields": {
            "sku": "sku",
            "name": "name",
            "price": "price",
            "description": "description",
            "category": "category_name",
            "attributes": "specs_json",
            "active": "is_active",
            "group_id": "family",
            "options": "opts",
        },
    }
    raw["entities"]["inventory"] = {
        "table": "variant_stock",
        "fields": {"sku": "sku", "quantity": "on_hand"},
    }
    adapter = SqlAdapter.from_url(
        sqlite_url, SchemaMapping.model_validate(raw), timezone="Asia/Ho_Chi_Minh"
    )
    yield BusinessService(adapter, app_config.business_rules)
    await adapter.close()


async def test_variants_of_one_product_are_one_search_result(variant_service: BusinessService):
    result = await variant_service.search_products(ALICE, "basic tee")
    products = result.data["products"]
    assert result.data["count"] == 1 and products[0]["name"] == "Basic tee"
    assert [(v["sku"], v["options"], v["price"]) for v in products[0]["variants"]] == [
        ("TEE-S", {"size": "S"}, 199_000),  # JSON text is parsed
        ("TEE-M", {"size": "M"}, 199_000),
        ("TEE-XL", {"size": "XL"}, 219_000),
    ]
    assert "options" not in products[0]


async def test_a_search_that_matches_one_variant_still_lists_its_siblings(
    variant_service: BusinessService,
):
    result = await variant_service.search_products(ALICE, "XL")  # only TEE-XL's options say XL
    hit = result.data["products"][0]
    assert hit["sku"] in {"TEE-S", "TEE-M", "TEE-XL"} and len(hit["variants"]) == 3


async def test_products_without_variants_are_unchanged(variant_service: BusinessService):
    (mug,) = (await variant_service.search_products(ALICE, "mug")).data["products"]
    assert mug["sku"] == "MUG-1" and "variants" not in mug and "options" not in mug


async def test_groups_count_once_against_the_result_limit(variant_service: BusinessService):
    result = await variant_service.search_products(ALICE, "tee mug cotton ceramic", limit=2)
    assert result.data["count"] == 2


async def test_stock_by_name_checks_every_variant(variant_service: BusinessService):
    items = (await variant_service.check_stock(ALICE, query="basic tee")).data["items"]
    assert {i["sku"]: (i["options"], i["status"]) for i in items} == {
        "TEE-S": ({"size": "S"}, "low_stock"),
        "TEE-M": ({"size": "M"}, "out_of_stock"),
        "TEE-XL": ({"size": "XL"}, "in_stock"),
    }


async def test_comparison_shows_which_variant_each_column_is(variant_service: BusinessService):
    result = await variant_service.compare_products(ALICE, ["TEE-S", "TEE-XL"])
    rows = {r["attribute"]: r["values"] for r in result.data["rows"]}
    assert rows["options"] == {"TEE-S": {"size": "S"}, "TEE-XL": {"size": "XL"}}


async def test_an_order_is_priced_per_variant(variant_service: BusinessService):
    from support_agent.mcp_db.service import OrderLineRequest

    ok = await variant_service.prepare_order_draft(
        ALICE, [OrderLineRequest(sku="TEE-XL", qty=2)], "12 Le Loi, District 1", "card"
    )
    assert ok.ok and ok.data["total"] == 2 * 219_000
    sold_out = await variant_service.prepare_order_draft(
        ALICE, [OrderLineRequest(sku="TEE-M", qty=1)], "12 Le Loi, District 1", "card"
    )
    assert sold_out.error.code == "OUT_OF_STOCK"
