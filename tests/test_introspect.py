"""`introspect-db`: a draft schema mapping read from a live database."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
import yaml
from mongomock_motor import AsyncMongoMockClient
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.mcp_db.adapters.mongo import MongoAdapter
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.introspect import (
    Column,
    Table,
    fold,
    is_shipped_template,
    propose,
    propose_mongo,
    read_sql,
    render_yaml,
    status_for,
    table_key,
    tables_from_documents,
)
from support_agent.mcp_db.mapping import MappingError, SchemaMapping, load_mapping
from support_agent.seed.demo import mongo_documents
from tests.conftest import DEMO, ROOT, TZ

# --- words ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Đã Giao", "da_giao"),
        ("DA_GIAO", "da_giao"),
        ("orderStatus", "order_status"),
        ("  Order-Total ", "order_total"),
    ],
)
def test_names_are_folded_to_one_spelling(raw: str, expected: str):
    assert fold(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("orders", "order"),
        ("public.Order_Items", "order_item"),
        ("tbl_customers", "customer"),
        ("wp_Products", "product"),
        ("categories", "category"),
        ("stock", "stock"),
        ("address", "address"),
    ],
)
def test_table_names_are_reduced_to_a_singular_key(raw: str, expected: str):
    assert table_key(raw) == expected


@pytest.mark.parametrize(
    ("entity", "raw", "expected"),
    [
        ("order", "DA_GIAO", "delivered"),
        ("order", "Đang giao", "shipping"),
        ("order", "CHO_XU_LY", "processing"),
        ("order", "HUY", "cancelled"),
        ("order", "unpaid", "pending_payment"),  # not "processing", though it contains "paid"
        ("order", "paid", "processing"),
        ("order", "shipped", "shipping"),
        ("order", "completed", "delivered"),
        ("order", "partially_shipped", "partially_shipped"),
        ("order", "on-hold", "on_hold"),
        ("order", "refunded", "refunded"),
        ("order", "mystery", None),
        ("shipment", "DANG_VAN_CHUYEN", "in_transit"),
        ("shipment", "CHO_LAY_HANG", "pending_pickup"),
        ("return_request", "TU_CHOI", "rejected"),
        ("return_request", "DA_DUYET", "approved"),
    ],
)
def test_status_values_are_recognised_in_both_languages(
    entity: str, raw: str, expected: str | None
):
    assert status_for(entity, raw) == expected


# --- the demo database: the answer is known ---------------------------------------------------------------


async def introspect_sql(url: str, dialect: str = "sqlite", **kw: Any):
    tables, values = await read_sql(url, schema=None, dialect=dialect)
    return propose(tables, dialect=dialect, currency="VND", status_values=values, **kw)


def fields(mapping_file: Path, entity: str) -> dict[str, str]:
    raw = yaml.safe_load(mapping_file.read_text(encoding="utf-8"))
    return {k: str(v) for k, v in raw["entities"][entity]["fields"].items()}


async def test_the_demo_database_is_mapped_the_way_a_person_mapped_it(sqlite_url: str):
    proposal = await introspect_sql(sqlite_url)
    shipped = DEMO / "config" / "schema_mapping.sqlite.yaml"
    assert proposal.complete and proposal.missing_entities == []
    for entity, guess in proposal.entities.items():
        found = {f: g.column for f, g in guess.fields.items()}
        expected = fields(shipped, entity)
        expected.pop("currency", None)  # a constant in the shipped file, absent from the table
        assert found == expected, entity
        assert (
            guess.table == yaml.safe_load(shipped.read_text("utf-8"))["entities"][entity]["table"]
        )


async def test_the_draft_loads_and_validates_against_the_database_it_came_from(
    sqlite_url: str, tmp_path: Path
):
    proposal = await introspect_sql(sqlite_url)
    path = tmp_path / "draft.yaml"
    path.write_text(render_yaml(proposal), encoding="utf-8")
    mapping = load_mapping(path)
    adapter = SqlAdapter.from_url(sqlite_url, mapping, timezone="Asia/Ho_Chi_Minh")
    try:
        assert await adapter.validate() == []
        order = await adapter.get_order("1234", "u_100")
        assert order and order["status"] == "delivered" and order["currency"] == "VND"
    finally:
        await adapter.close()


async def test_status_values_come_from_the_data_and_the_draft_says_so(sqlite_url: str):
    proposal = await introspect_sql(sqlite_url)
    order = proposal.entities["order"]
    assert order.status_map == {
        "CHO_XU_LY": "processing",
        "DANG_GIAO": "shipping",
        "DA_GIAO": "delivered",
        "HUY": "cancelled",
    }
    assert "the values found today" in render_yaml(proposal)


async def test_a_missing_currency_column_becomes_a_labelled_constant(sqlite_url: str):
    proposal = await introspect_sql(sqlite_url)
    text = render_yaml(proposal)
    assert """currency: "'VND'"  # no column: a constant""" in text
    assert any("no currency column" in n for n in proposal.notes)


# --- a different schema: English names, ids, foreign keys --------------------------------------------------


async def make_foreign_shop(tmp_path: Path, *, with_fks: bool = True) -> str:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'other.db').as_posix()}"
    md = sa.MetaData()
    fk = (lambda target: sa.ForeignKey(target)) if with_fks else (lambda target: None)

    def col(name: str, type_: Any, *refs: Any, **kw: Any) -> sa.Column:
        return sa.Column(name, type_, *[r for r in refs if r is not None], **kw)

    sa.Table(
        "Users",
        md,
        col("id", sa.Integer, primary_key=True),
        col("email_address", sa.String(100)),
        col("mobile", sa.String(20)),
        col("display_name", sa.String(100)),
    )
    sa.Table(
        "Orders",
        md,
        col("id", sa.Integer, primary_key=True),
        col("order_number", sa.String(20)),  # what the customer sees, but NOT what lines point at
        col("user_id", sa.Integer, fk("Users.id")),
        col("state", sa.String(20)),
        col("total_amount", sa.Numeric(10, 2)),
        col("discount_total", sa.Numeric(10, 2)),
        col("shipping_total", sa.Numeric(10, 2)),
        col("placed_at", sa.DateTime),
        col("currency_code", sa.String(3)),
    )
    sa.Table(
        "Products",
        md,
        col("id", sa.Integer, primary_key=True),
        col("sku", sa.String(30)),
        col("title", sa.String(100)),
        col("regular_price", sa.Numeric(10, 2)),
    )
    sa.Table(
        "OrderItems",
        md,
        col("id", sa.Integer, primary_key=True),
        col("order_id", sa.Integer, fk("Orders.id")),
        col("product_id", sa.Integer, fk("Products.id")),
        col("qty", sa.Integer),
        col("item_price", sa.Numeric(10, 2)),
    )
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(md.create_all)
        await conn.execute(
            sa.text(
                "INSERT INTO Orders (id, order_number, user_id, state, total_amount, placed_at) "
                "VALUES (1, 'A-1', 7, 'wc-completed', 10, '2026-01-01 10:00:00'), "
                "(2, 'A-2', 7, 'wc-on-hold', 5, '2026-01-02 10:00:00'), "
                "(3, 'A-3', 7, 'frobnicated', 5, '2026-01-03 10:00:00')"
            )
        )
    await engine.dispose()
    return url


async def test_a_foreign_schema_is_read_through_its_foreign_keys(tmp_path: Path):
    proposal = await introspect_sql(await make_foreign_shop(tmp_path))
    order = {f: g.column for f, g in proposal.entities["order"].fields.items()}
    # Lines point at Orders.id, so that is "the order id" even though order_number exists.
    assert order["id"] == "id"
    assert order["customer_id"] == "user_id" and order["status"] == "state"
    assert order["total"] == "total_amount" and order["created_at"] == "placed_at"
    assert order["currency"] == "currency_code"
    assert order["discount"] == "discount_total" and order["shipping_fee"] == "shipping_total"
    assert proposal.entities["customer"].fields["id"].column == "id"
    assert proposal.entities["customer"].fields["email"].column == "email_address"
    assert proposal.entities["customer"].fields["phone"].column == "mobile"
    item = {f: g.column for f, g in proposal.entities["order_item"].fields.items()}
    assert item["order_id"] == "order_id" and item["quantity"] == "qty"
    assert item["unit_price"] == "item_price"


async def test_without_foreign_keys_the_same_names_still_line_up(tmp_path: Path):
    proposal = await introspect_sql(await make_foreign_shop(tmp_path, with_fks=False))
    assert proposal.entities["order"].fields["id"].column == "id"
    assert proposal.entities["order"].fields["customer_id"].column == "user_id"


async def test_lines_that_point_at_a_surrogate_product_id_are_reported(tmp_path: Path):
    proposal = await introspect_sql(await make_foreign_shop(tmp_path))
    assert any("create a view" in p and "product_id" in p for p in proposal.problems)
    assert not proposal.complete


async def test_unknown_statuses_are_left_for_a_person_not_guessed(tmp_path: Path):
    proposal = await introspect_sql(await make_foreign_shop(tmp_path))
    order = proposal.entities["order"]
    assert order.status_map == {"wc-completed": "delivered", "wc-on-hold": "on_hold"}
    assert order.unmapped_statuses == ["frobnicated"]
    assert "# frobnicated: ???" in render_yaml(proposal)
    assert not proposal.complete


async def test_required_fields_that_are_not_found_fail_loudly(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'odd.db').as_posix()}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.execute(sa.text("CREATE TABLE orders (id INTEGER, who TEXT, what TEXT)"))
        await conn.execute(sa.text("CREATE TABLE customers (id INTEGER, email TEXT)"))
    await engine.dispose()
    proposal = await introspect_sql(url)
    assert "order_item" in proposal.missing_entities
    assert {"customer_id", "status", "total", "created_at"} <= set(
        proposal.entities["order"].missing_required
    )
    text = render_yaml(proposal)
    assert "TODO_order_item" in text and "TODO_order_total  # required; not found" in text
    path = tmp_path / "draft.yaml"
    path.write_text(text, encoding="utf-8")
    load_mapping(path)  # it still parses, so the owner can fix it in place...
    mapping = load_mapping(path)
    adapter = SqlAdapter.from_url(url, mapping)
    try:
        assert any(i.level == "error" for i in await adapter.validate())  # ...but it never passes
    finally:
        await adapter.close()


def test_a_database_without_the_shop_tables_proposes_nothing_silly():
    tables = [Table("alembic_version", [Column("version_num")]), Table("sessions", [Column("id")])]
    proposal = propose(tables, dialect="sqlite", currency="VND")
    assert proposal.entities == {}
    assert set(proposal.missing_entities) == {"customer", "order", "order_item"}
    assert proposal.unused_tables == ["alembic_version", "sessions"]


def test_a_table_is_recognised_by_its_columns_when_its_name_says_nothing():
    cols = [
        Column(n, k)
        for n, k in [
            ("id", "number"),
            ("customer_id", "number"),
            ("status", "text"),
            ("total", "number"),
            ("created_at", "date"),
        ]
    ]
    proposal = propose([Table("tx_log", cols)], dialect="sqlite", currency="VND")
    assert proposal.entities["order"].table == "tx_log"
    assert proposal.entities["order"].confidence == "check"
    assert "check: matched by its columns" in render_yaml(proposal)


def test_a_refund_column_is_not_taken_for_the_order_total():
    cols = [Column("refund_amount", "number"), Column("subtotal", "number")]
    t = Table("orders", cols)
    proposal = propose([t], dialect="sqlite", currency="VND")
    assert "total" not in proposal.entities.get("order", type("E", (), {"fields": {}})).fields


def test_a_numeric_field_prefers_a_numeric_column_over_a_text_one():
    cols = [
        Column("id", "number"),
        Column("customer_id", "number"),
        Column("status", "text"),
        Column("total", "text"),
        Column("amount", "number"),
        Column("created_at", "date"),
    ]
    proposal = propose([Table("orders", cols)], dialect="sqlite", currency="VND")
    assert proposal.entities["order"].fields["total"].column == "amount"


# --- PostgreSQL naming and the shipped template ---------------------------------------------------------------


def test_postgres_tables_keep_their_schema_in_the_draft():
    cols = [Column("id", "number"), Column("email")]
    proposal = propose([Table("public.customers", cols)], dialect="postgres", currency="VND")
    assert "table: public.customers" in render_yaml(proposal)


def test_the_shipped_template_is_recognised_so_the_first_write_needs_no_force(tmp_path: Path):
    assert is_shipped_template(ROOT / "config" / "schema_mapping.yaml")
    edited = tmp_path / "mine.yaml"
    edited.write_text("dialect: sqlite\nentities: {}\n", encoding="utf-8")
    assert not is_shipped_template(edited)
    assert not is_shipped_template(tmp_path / "missing.yaml")


# --- MongoDB -------------------------------------------------------------------------------------------------------------------


async def test_mongodb_embedded_lines_and_shipment_are_found():
    db = AsyncMongoMockClient(tz_aware=True)["shop"]
    docs = mongo_documents(datetime.now(TZ))
    for name, items in docs.items():
        await db[name].insert_many(items)
    samples = {n: await db[n].find({}).to_list(100) for n in await db.list_collection_names()}

    values = {
        ("orders", "status"): ["CHO_XU_LY", "DANG_GIAO", "DA_GIAO", "HUY"],
        ("orders", "shipment.status"): ["DANG_VAN_CHUYEN", "DA_GIAO"],
        ("return_requests", "status"): ["CHO_DUYET"],
    }
    proposal = propose_mongo(
        tables_from_documents(samples),
        currency="VND",
        status_values=lambda table, column: values.get((table, column), []),
    )
    shipped = yaml.safe_load((DEMO / "config" / "schema_mapping.mongodb.yaml").read_text("utf-8"))
    for entity, guess in proposal.entities.items():
        expected = {k: str(v) for k, v in shipped["entities"][entity]["fields"].items()}
        expected.pop("currency", None)
        assert {f: g.column for f, g in guess.fields.items()} == expected, entity
        assert guess.table == shipped["entities"][entity]["collection"]
    assert proposal.entities["order_item"].unwind == "items"
    assert proposal.entities["shipment"].status_map["DANG_VAN_CHUYEN"] == "in_transit"

    text = render_yaml(proposal)
    mapping = SchemaMapping.model_validate(yaml.safe_load(text))
    assert mapping.dialect == "mongodb"
    adapter = MongoAdapter(db, mapping, timezone="Asia/Ho_Chi_Minh")
    order = await adapter.get_order("1234", "u_100")
    assert order and order["status"] == "delivered"
    items = await adapter.get_order_items("1234")
    assert items and items[0]["sku"] == "EAR-BT20"
    ship = await adapter.get_shipment("1236")
    assert ship and ship["status"] == "in_transit"


def test_an_unfilled_draft_is_rejected_by_the_mapping_validator_when_it_is_really_wrong():
    bad = "dialect: sqlite\nentities:\n  customer:\n    table: 'x; DROP'\n    fields: {id: id}\n"
    with pytest.raises(MappingError):
        load_mapping_text(bad)


def load_mapping_text(text: str) -> SchemaMapping:
    from pydantic import ValidationError

    try:
        return SchemaMapping.model_validate(yaml.safe_load(text))
    except ValidationError as exc:
        raise MappingError(str(exc)) from exc
