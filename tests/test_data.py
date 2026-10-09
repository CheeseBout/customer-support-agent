from __future__ import annotations

import asyncio
import logging
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
import yaml
from mongomock_motor import AsyncMongoMockClient
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.core.principal import sign_principal
from support_agent.core.settings import Settings
from support_agent.mcp_db.adapters.base import AdapterError, AdapterTimeout, DataAdapter
from support_agent.mcp_db.adapters.mongo import MongoAdapter
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.factory import ConfigError, create_adapter, normalise_db_url
from support_agent.mcp_db.mapping import (
    Call,
    Col,
    Lit,
    MappingError,
    SchemaMapping,
    load_mapping,
    parse_expression,
)
from support_agent.mcp_db.server import PRINCIPAL_META_KEY, build_server
from support_agent.mcp_db.service import BusinessService, fold
from support_agent.seed.demo import mongo_documents
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, DEMO, SECRET, STAFF, TZ

EXAMPLES = DEMO / "config"

# --- expression parser --------------------------------------------------------------------------


def test_parse_expressions():
    assert parse_expression("order_code") == Col("order_code")
    assert parse_expression("items.sku") == Col("items.sku")
    assert parse_expression("'VND'") == Lit("VND")
    assert parse_expression("'it''s'") == Lit("it's")
    assert parse_expression("42") == Lit(42)
    assert parse_expression("COALESCE(a, 'x')") == Call("COALESCE", (Col("a"), Lit("x")))
    nested = parse_expression("concat(first, ' ', COALESCE(middle, last))")
    assert nested.name == "CONCAT" and len(nested.args) == 3


@pytest.mark.parametrize(
    "bad",
    [
        "LOWER(name)",
        "a; DROP TABLE users",
        "a + b",
        "(a)",
        "COALESCE(a",
        "'unterminated",
        "a b",
        "",
        "UNION SELECT 1",
    ],
)
def test_parser_rejects_anything_outside_the_whitelist(bad: str):
    with pytest.raises(MappingError):
        parse_expression(bad)


# --- mapping validation -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    sorted(EXAMPLES.glob("schema_mapping.*.yaml")),
    ids=lambda p: p.name,
)
def test_shipped_mappings_are_valid(path: Path):
    m = load_mapping(path)
    assert m.entity("order").status_map["DA_GIAO"] == "delivered"


def mapping_dict() -> dict[str, Any]:
    return yaml.safe_load((EXAMPLES / "schema_mapping.sqlite.yaml").read_text(encoding="utf-8"))


def _load(d: dict[str, Any]) -> SchemaMapping:
    path = Path(tempfile.mkdtemp()) / "m.yaml"
    path.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    return load_mapping(path)


def test_mapping_reports_every_problem_at_once():
    d = mapping_dict()
    del d["entities"]["order_item"]
    del d["entities"]["order"]["fields"]["customer_id"]
    d["entities"]["order"]["fields"]["colour"] = "colour"
    d["entities"]["product"]["fields"]["name"] = "LOWER(name)"
    d["entities"]["order"]["status_map"]["X"] = "teleported"
    with pytest.raises(MappingError) as exc:
        _load(d)
    text = str(exc.value)
    for fragment in (
        "missing required entity 'order_item'",
        "order: missing required field 'customer_id'",
        "order: unknown field 'colour'",
        "not allowed",
        "unknown canonical status",
    ):
        assert fragment in text


def test_sql_mapping_rejects_unsafe_tables_and_dotted_columns():
    d = mapping_dict()
    d["entities"]["order"]["table"] = "orders; DROP TABLE orders"
    d["entities"]["customer"]["fields"]["email"] = "profile.email"
    with pytest.raises(MappingError) as exc:
        _load(d)
    assert "unsafe table name" in str(exc.value) and "dotted column" in str(exc.value)


def test_mongo_mapping_needs_collections():
    d = yaml.safe_load((EXAMPLES / "schema_mapping.mongodb.yaml").read_text(encoding="utf-8"))
    del d["entities"]["order"]["collection"]
    with pytest.raises(MappingError, match="need `collection`"):
        _load(d)


def test_missing_mapping_file(tmp_path: Path):
    with pytest.raises(MappingError, match="not found"):
        load_mapping(tmp_path / "nope.yaml")


def test_factory_checks_dialect_and_url(settings: Settings, sqlite_mapping: SchemaMapping):
    s = settings.model_copy(
        update={"business_db_type": "postgres", "business_db_url": "postgresql://x"}
    )
    with pytest.raises(ConfigError, match="dialect"):
        create_adapter(s, sqlite_mapping)
    with pytest.raises(ConfigError, match="BUSINESS_DB_URL"):
        create_adapter(settings.model_copy(update={"business_db_url": None}), sqlite_mapping)
    assert normalise_db_url("postgres", "postgresql://u:p@h/db") == "postgresql+asyncpg://u:p@h/db"
    assert normalise_db_url("mysql", "mysql://u:p@h/db") == "mysql+aiomysql://u:p@h/db"
    assert normalise_db_url("sqlite", "sqlite:///a.db") == "sqlite+aiosqlite:///a.db"
    assert normalise_db_url("postgres", "postgresql+asyncpg://x") == "postgresql+asyncpg://x"


# --- adapter contract: identical behaviour on SQL and MongoDB ----------------------------------------


@pytest_asyncio.fixture(params=["sql", "mongo"])
async def any_adapter(
    request: pytest.FixtureRequest, sqlite_url: str, sqlite_mapping: SchemaMapping
) -> AsyncIterator[DataAdapter]:
    if request.param == "sql":
        a: DataAdapter = SqlAdapter.from_url(
            sqlite_url, sqlite_mapping, timezone="Asia/Ho_Chi_Minh"
        )
    else:
        db = AsyncMongoMockClient(tz_aware=True)["shop"]
        for name, docs in mongo_documents(datetime.now(TZ)).items():
            await db[name].insert_many(docs)
        a = MongoAdapter(
            db, load_mapping(EXAMPLES / "schema_mapping.mongodb.yaml"), timezone="Asia/Ho_Chi_Minh"
        )
    yield a
    await a.close()


async def test_get_order_enforces_ownership_in_the_query(any_adapter: DataAdapter):
    own = await any_adapter.get_order("1234", "u_100")
    assert own and own["id"] == "1234" and own["status"] == "delivered" and own["total"] == 350_000
    assert own["currency"] == "VND" and own["delivered_at"].endswith(("+07:00", "+00:00"))
    assert await any_adapter.get_order("1234", "u_101") is None  # someone else's
    assert await any_adapter.get_order("0000", "u_100") is None  # nonexistent
    assert (await any_adapter.get_order("1234", None))["id"] == "1234"  # staff path


async def test_status_map_translates_db_codes(any_adapter: DataAdapter):
    statuses = {o["id"]: o["status"] for o in await any_adapter.list_orders("u_100", None, 10)}
    assert statuses["1234"] == "delivered" and statuses["1236"] == "shipping"
    assert statuses["1237"] == "processing" and statuses["1238"] == "cancelled"


async def test_list_orders_filters_sorts_and_limits(any_adapter: DataAdapter):
    delivered = await any_adapter.list_orders("u_100", "delivered", 10)
    assert {o["id"] for o in delivered} == {"1234", "1235", "1239", "1240"}
    dates = [o["created_at"] for o in delivered]
    assert dates == sorted(dates, reverse=True)
    assert len(await any_adapter.list_orders("u_100", None, 2)) == 2
    assert {o["id"] for o in await any_adapter.list_orders("u_101", None, 10)} == {"2001", "2002"}
    assert await any_adapter.list_orders("nobody", None, 10) == []


async def test_order_items_shipments_returns_inventory_products(any_adapter: DataAdapter):
    items = await any_adapter.get_order_items("2001")
    assert {i["sku"] for i in items} == {"PHN-X100", "CASE-X100"}
    assert all(isinstance(i["quantity"], int) and isinstance(i["unit_price"], int) for i in items)

    ship = await any_adapter.get_shipment("1236")
    assert ship["status"] == "in_transit" and ship["carrier"] == "GHTK" and ship["eta"]
    none_yet = await any_adapter.get_shipment("1237")
    assert not none_yet or all(v is None for k, v in none_yet.items() if k != "order_id")

    reqs = await any_adapter.get_return_requests("1240")
    assert [(r["status"], r["sku"]) for r in reqs] == [("pending", "SPK-BM5")]
    assert await any_adapter.get_return_requests("1234") == []

    assert await any_adapter.get_inventory(["LAP-PRO14", "CHG-65W", "GHOST"]) == {
        "LAP-PRO14": 3,
        "CHG-65W": 0,
    }
    assert await any_adapter.get_inventory([]) == {}

    products = await any_adapter.get_products(["PHN-X100"])
    assert products["PHN-X100"]["attributes"]["battery_mah"] == 5000  # JSON text parsed
    assert products["PHN-X100"]["active"] is True


async def test_product_candidates_respect_sql_side_filters(any_adapter: DataAdapter):
    cheap = await any_adapter.search_products(
        category=None, min_price=None, max_price=300_000, candidate_cap=100
    )
    assert cheap and all(p["price"] <= 300_000 for p in cheap)
    acc = await any_adapter.search_products(
        category="Accessories", min_price=None, max_price=None, candidate_cap=100
    )
    assert acc and {p["category"] for p in acc} == {"accessories"}


async def test_validate_passes_on_a_correct_mapping(any_adapter: DataAdapter):
    assert await any_adapter.validate() == []


# --- SQL specifics ------------------------------------------------------------------------------------


async def test_sql_adapter_is_read_only(adapter: SqlAdapter):
    async with adapter.engine.connect() as conn:
        with pytest.raises(sa.exc.DBAPIError):
            await conn.execute(sa.text("DELETE FROM orders"))


async def test_row_cap_is_enforced(sqlite_url: str, sqlite_mapping: SchemaMapping):
    a = SqlAdapter.from_url(sqlite_url, sqlite_mapping, max_rows=2)
    try:
        assert len(await a.list_orders("u_100", None, 10)) == 2
        assert len(await a.get_order_items("2001")) == 2
    finally:
        await a.close()


async def test_query_timeout_maps_to_adapter_timeout(
    adapter: SqlAdapter, monkeypatch: pytest.MonkeyPatch
):
    adapter.timeout_seconds = 0.05

    class SlowEngine:
        @asynccontextmanager
        async def connect(self):
            await asyncio.sleep(1)
            yield None

    monkeypatch.setattr(adapter, "engine", SlowEngine())
    with pytest.raises(AdapterTimeout):
        await adapter.get_order("1234", "u_100")


async def test_driver_errors_do_not_leak_sql(sqlite_url: str, sqlite_mapping: SchemaMapping):
    d = mapping_dict()
    d["entities"]["order"]["fields"]["status"] = "no_such_column"
    a = SqlAdapter.from_url(sqlite_url, _load(d))
    try:
        with pytest.raises(AdapterError) as exc:
            await a.get_order("1234", "u_100")
        assert "no_such_column" not in str(exc.value) and "SELECT" not in str(exc.value)
    finally:
        await a.close()


async def test_sql_parameters_are_bound_not_interpolated(adapter: SqlAdapter):
    evil = "1234' OR '1'='1"
    assert await adapter.get_order(evil, "u_100") is None
    assert await adapter.get_order("1234", "u_100' OR '1'='1") is None
    assert await adapter.list_orders("u_100' OR '1'='1", None, 10) == []


async def test_validate_reports_mapping_problems(sqlite_url: str):
    d = mapping_dict()
    d["entities"]["order"]["fields"]["total"] = "nonexistent_total"
    d["entities"]["shipment"]["table"] = "missing_table"
    d["entities"]["customer"]["fields"]["name"] = "id"  # valid column, plain text: fine
    a = SqlAdapter.from_url(sqlite_url, _load(d))
    try:
        messages = " | ".join(i.message for i in await a.validate())
    finally:
        await a.close()
    assert "nonexistent_total" in messages and "missing_table" in messages


async def test_validate_flags_status_values_missing_from_status_map(
    sqlite_url: str, sqlite_mapping: SchemaMapping
):
    engine = create_async_engine(sqlite_url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE orders SET status = 'LA_LA' WHERE order_code = '1237'"))
    await engine.dispose()
    a = SqlAdapter.from_url(sqlite_url, sqlite_mapping)
    try:
        issues = await a.validate()
    finally:
        await a.close()
    assert [(i.level, i.entity) for i in issues] == [("error", "order")]
    assert "LA_LA" in issues[0].message


async def test_integer_ids_match_string_input(tmp_path: Path, sqlite_mapping: SchemaMapping):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'n.db').as_posix()}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(
            sa.text(
                "CREATE TABLE orders (order_code INTEGER, customer_id INTEGER, status TEXT, grand_total INTEGER, created_at TEXT, delivered_at TEXT, ship_address TEXT, pay_method TEXT)"
            )
        )
        await conn.execute(
            sa.text(
                "INSERT INTO orders VALUES (1234, 7, 'DA_GIAO', 100, '2026-10-01 10:00:00', NULL, 'x', 'cod')"
            )
        )
    await engine.dispose()
    a = SqlAdapter.from_url(url, sqlite_mapping)
    try:
        order = await a.get_order("1234", "7")
        assert order and order["id"] == "1234" and order["customer_id"] == "7"
    finally:
        await a.close()


# --- service -------------------------------------------------------------------------------------------------


async def test_not_yours_and_nonexistent_look_identical(service: BusinessService):
    theirs = await service.get_order(ALICE, "2001")
    ghost = await service.get_order(ALICE, "9999")
    assert theirs.error.code == ghost.error.code == "NOT_FOUND"
    assert theirs.error.message.replace("2001", "X") == ghost.error.message.replace("9999", "X")
    assert not (await service.get_shipment_status(ALICE, "2002")).ok
    assert not (await service.check_return_eligibility(ALICE, "2001")).ok


async def test_staff_can_read_any_order_customers_cannot(service: BusinessService):
    assert (await service.get_order(STAFF, "2001")).ok
    assert (await service.get_order(BOB, "2001")).ok
    assert not (await service.get_order(BOB, "1234")).ok


async def test_get_order_hides_customer_id_and_includes_items(service: BusinessService):
    r = await service.get_order(ALICE, "1234")
    assert "customer_id" not in r.data["order"]
    assert r.data["items"][0]["sku"] == "EAR-BT20"


async def test_list_orders_clamps_limit_and_scopes_to_caller(service: BusinessService):
    r = await service.list_orders(ALICE, limit=500)
    assert r.ok and 0 < r.data["count"] <= 10
    assert {o["id"] for o in r.data["orders"]} <= {
        "1234",
        "1235",
        "1236",
        "1237",
        "1238",
        "1239",
        "1240",
    }
    assert (await service.list_orders(ALICE, limit=0)).data["count"] == 1
    assert (await service.list_orders(ALICE, status="shipping")).data["orders"][0]["id"] == "1236"


async def test_shipment_status(service: BusinessService):
    r = await service.get_shipment_status(ALICE, "1236")
    assert r.data["shipment"]["carrier"] == "GHTK" and "order_id" not in r.data["shipment"]
    none = await service.get_shipment_status(ALICE, "1237")
    assert none.ok and none.data["shipment"] is None and "No shipment" in none.data["note"]


async def test_stock_hides_exact_quantity_unless_configured_or_staff(
    service: BusinessService, monkeypatch: pytest.MonkeyPatch
):
    r = await service.check_stock(ALICE, sku="PHN-X100")
    assert r.data["items"] == [
        {"sku": "PHN-X100", "name": "Điện thoại Nova X100", "status": "in_stock"}
    ]
    statuses = {
        i["sku"]: i["status"]
        for i in (await service.check_stock(ALICE, query="laptop")).data["items"]
    }
    assert statuses["LAP-PRO14"] == "low_stock"  # 3 <= threshold 5
    assert (await service.check_stock(ALICE, sku="CHG-65W")).data["items"][0][
        "status"
    ] == "out_of_stock"
    assert (await service.check_stock(STAFF, sku="LAP-PRO14")).data["items"][0]["quantity"] == 3

    monkeypatch.setattr(service.rules.inventory, "show_exact_quantity", True)  # undone after
    assert (await service.check_stock(ALICE, sku="LAP-PRO14")).data["items"][0]["quantity"] == 3


async def test_stock_argument_errors(service: BusinessService):
    assert (await service.check_stock(ALICE)).error.code == "INVALID_ARGUMENT"
    assert (await service.check_stock(ALICE, sku="GHOST-1")).error.code == "NOT_FOUND"
    assert (await service.check_stock(ALICE, query="zzzzqqq")).error.code == "NOT_FOUND"


async def test_product_search_is_accent_and_language_insensitive(service: BusinessService):
    for query in ("dien thoai pin lau", "điện thoại", "phone battery", "PHONE"):
        r = await service.search_products(ALICE, query)
        assert r.data["products"] and r.data["products"][0]["sku"] == "PHN-X100", query


async def test_product_search_filters_limit_and_empty_query(service: BusinessService):
    from support_agent.mcp_db.service import ProductFilters

    r = await service.search_products(ALICE, "laptop", ProductFilters(max_price=15_000_000))
    assert [p["sku"] for p in r.data["products"]] == ["LAP-AIR13"]
    assert (await service.search_products(ALICE, "laptop", limit=1)).data["count"] == 1
    assert (await service.search_products(ALICE, "the and of")).error.code == "INVALID_ARGUMENT"
    r = await service.search_products(ALICE, "cap usb", ProductFilters(category="accessories"))
    assert r.data["products"][0]["sku"] == "CAB-USBC2"


async def test_inactive_products_are_hidden(service: BusinessService, adapter: SqlAdapter):
    engine = create_async_engine(str(adapter.engine.url))
    async with engine.begin() as conn:
        await conn.execute(sa.text("UPDATE products SET is_active = 0 WHERE sku = 'PHN-X100'"))
    await engine.dispose()
    assert not (await service.search_products(ALICE, "phone")).data["products"]
    assert (await service.check_stock(ALICE, sku="PHN-X100")).error.code == "NOT_FOUND"


async def test_long_descriptions_are_truncated(service: BusinessService):
    from support_agent.mcp_db import service as svc

    view = BusinessService._product_view({"sku": "X", "name": "n", "description": "d" * 1000})
    assert len(view["description"]) == svc.DESCRIPTION_CHARS + 1 and view["description"].endswith(
        "…"
    )


async def test_eligibility_scenarios_on_demo_data(service: BusinessService):
    async def run(order_id: str) -> dict[str, Any]:
        return (await service.check_return_eligibility(ALICE, order_id)).data

    ok = await run("1234")
    assert ok["eligible"] and ok["refundable_amount"] == 350_000 and ok["days_remaining"] in (2, 3)
    assert (await run("1235"))["reasons"] == ["WINDOW_EXPIRED"]
    assert (await run("1239"))["reasons"] == ["CATEGORY_EXCLUDED"]
    assert (await run("1240"))["reasons"] == ["ALREADY_REQUESTED"]
    assert set((await run("1238"))["reasons"]) == {"STATUS_NOT_ALLOWED", "WINDOW_EXPIRED"}
    assert (await service.check_return_eligibility(ALICE, "1234", sku="NOPE-1")).data[
        "reasons"
    ] == ["ITEM_NOT_FOUND"]


async def test_adapter_failures_become_standard_errors(
    service: BusinessService, monkeypatch: pytest.MonkeyPatch
):
    async def timeout(*a: Any, **k: Any) -> None:
        raise AdapterTimeout("slow")

    async def broken(*a: Any, **k: Any) -> None:
        raise AdapterError("down")

    monkeypatch.setattr(service.adapter, "get_order", timeout)
    assert (await service.get_order(ALICE, "1234")).error.code == "TIMEOUT"
    monkeypatch.setattr(service.adapter, "get_order", broken)
    r = await service.get_order(ALICE, "1234")
    assert r.error.code == "UPSTREAM_ERROR" and "down" not in r.error.message


def test_fold_removes_diacritics():
    assert fold("Điện thoại Nova") == "dien thoai nova"


# --- MCP server: identity and argument handling ----------------------------------------------------------------


async def test_tools_work_end_to_end_over_mcp(tool_client: DomainToolClient):
    r = await tool_client.call("get_order", {"order_id": "1234"}, ALICE)
    assert r.ok and r.data["order"]["id"] == "1234"
    assert (
        await tool_client.call("get_order", {"order_id": "2001"}, ALICE)
    ).error.code == "NOT_FOUND"


async def test_identity_arguments_are_refused(
    tool_client: DomainToolClient, caplog: pytest.LogCaptureFixture
):
    for key in ("customer_id", "user_id", "principal", "role"):
        r = await tool_client.call("get_order", {"order_id": "2001", key: "u_101"}, ALICE)
        assert r.error.code == "FORBIDDEN", key
    assert "identity argument" in caplog.text


async def test_unknown_arguments_are_rejected_not_ignored(tool_client: DomainToolClient):
    r = await tool_client.call("list_orders", {"limit": 2, "sort": "asc"}, ALICE)
    assert r.error.code == "INVALID_ARGUMENT" and "sort" in r.error.message


async def test_mcp_level_type_errors_become_invalid_argument(tool_client: DomainToolClient):
    assert (await tool_client.call("get_order", {}, ALICE)).error.code == "INVALID_ARGUMENT"
    assert (
        await tool_client.call("get_order", {"order_id": 5}, ALICE)
    ).error.code == "INVALID_ARGUMENT"


async def test_forged_expired_and_missing_principals_are_forbidden(service: BusinessService):
    server = build_server(service, SECRET)
    async with DomainToolClient(server, b"another-secret") as forged:
        assert (
            await forged.call("get_order", {"order_id": "1234"}, ALICE)
        ).error.code == "FORBIDDEN"

    from mcp import Client

    async with Client(server) as raw:
        res = await raw.call_tool("get_order", {"order_id": "1234"})  # no principal at all
        assert '"FORBIDDEN"' in res.content[0].text
        expired = sign_principal(ALICE, SECRET, ttl_seconds=-5)
        res = await raw.call_tool(
            "get_order", {"order_id": "1234"}, meta={PRINCIPAL_META_KEY: expired}
        )
        assert '"FORBIDDEN"' in res.content[0].text
        swapped = sign_principal(BOB, SECRET)  # a valid signature for a different user is fine...
        res = await raw.call_tool(
            "get_order", {"order_id": "1234"}, meta={PRINCIPAL_META_KEY: swapped}
        )
        assert '"NOT_FOUND"' in res.content[0].text  # ...but it only ever sees that user's data


async def test_every_tool_call_is_audited_with_masked_args(
    tool_client: DomainToolClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.INFO, logger="support_agent.audit")
    await tool_client.call("search_products", {"query": "call 0901234567"}, ALICE)
    await tool_client.call("get_order", {"order_id": "2001"}, ALICE)
    records = [r for r in caplog.records if r.name == "support_agent.audit"]
    assert [r.tool for r in records] == ["search_products", "get_order"]
    assert records[1].result_code == "NOT_FOUND" and records[0].result_code == "OK"
    assert "0901234567" not in records[0].tool_args and "u_100" not in str(records[0].__dict__)
    assert all(isinstance(r.duration_ms, int) for r in records)


async def test_server_lists_only_domain_tools_without_identity_parameters(service: BusinessService):
    from mcp import Client

    async with Client(build_server(service, SECRET)) as raw:
        tools = (await raw.list_tools()).tools
    names = {t.name for t in tools}
    assert names == {
        "get_order",
        "list_orders",
        "get_shipment_status",
        "check_stock",
        "search_products",
        "check_return_eligibility",
        "check_warranty_eligibility",
        "compare_products",
        "prepare_order_draft",
    }
    for t in tools:
        props = set(t.input_schema.get("properties", {}))
        assert not props & {"customer_id", "user_id", "principal", "role", "ctx"}, t.name


def test_a_name_the_customer_uses_is_matched_to_one_line_of_the_order():
    from support_agent.mcp_db.service import match_order_sku

    items = [
        {"sku": "SPK-BM5", "product_name": "Loa Bluetooth Boom 5"},
        {"sku": "CAB-USBC2", "product_name": "Cáp USB-C 2m"},
    ]
    assert match_order_sku(items, "SPK-BM5") == "SPK-BM5"
    assert match_order_sku(items, "spk-bm5") == "SPK-BM5"
    assert match_order_sku(items, "Boom 5") == "SPK-BM5"
    assert match_order_sku(items, "cap usb-c") == "CAB-USBC2"  # accent-insensitive
    assert match_order_sku(items, "USB") == "CAB-USBC2"
    assert match_order_sku(items, "laptop") == "laptop"  # unknown: left for ITEM_NOT_FOUND
    assert match_order_sku(items, "a") == "a"  # matches both lines: ambiguous, left alone
    assert match_order_sku(items, None) is None
