"""PLAN Phase 2 / SPEC 18: the MCP data layer against real Postgres, MySQL and MongoDB.

The adapter-contract tests are imported from the unit suite so *exactly the same assertions*
run against every real database.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.core.settings import AppConfig
from support_agent.mcp_db.adapters.base import DataAdapter
from support_agent.mcp_db.server import build_server
from support_agent.mcp_db.service import BusinessService
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB, SECRET
from tests.integration.conftest import Deployment

# Re-exported so pytest collects them in this module with the real-database `any_adapter`.
from tests.test_data import (  # noqa: F401
    test_get_order_enforces_ownership_in_the_query,
    test_list_orders_filters_sorts_and_limits,
    test_order_items_shipments_returns_inventory_products,
    test_product_candidates_respect_sql_side_filters,
    test_status_map_translates_db_codes,
    test_validate_passes_on_a_correct_mapping,
)

pytestmark = pytest.mark.integration


async def test_adapter_rejects_injection_attempts(any_adapter: DataAdapter):
    assert await any_adapter.get_order("1234' OR '1'='1", "u_100") is None
    assert await any_adapter.list_orders("u_100' OR '1'='1", None, 10) == []


async def test_row_limit_holds_on_a_real_database(any_adapter: DataAdapter):
    any_adapter.max_rows = 2
    assert len(await any_adapter.list_orders("u_100", None, 10)) == 2


async def test_readonly_account_cannot_write(deployment: Deployment):
    if deployment.kind == "mongodb":
        pytest.skip("read-only Mongo role is not provisioned in the test container")
    engine = create_async_engine(deployment.readonly_url)
    try:
        async with engine.begin() as conn:
            with pytest.raises(sa.exc.DBAPIError):
                await conn.execute(sa.text("DELETE FROM orders"))
    finally:
        await engine.dispose()


async def test_end_to_end_tools_and_isolation(any_adapter: DataAdapter, app_config: AppConfig):
    service = BusinessService(any_adapter, app_config.business_rules)
    async with DomainToolClient(build_server(service, SECRET), SECRET) as client:
        mine = await client.call("get_order", {"order_id": "1234"}, ALICE)
        assert mine.ok and mine.data["items"][0]["sku"] == "EAR-BT20"
        assert (
            await client.call("get_order", {"order_id": "2001"}, ALICE)
        ).error.code == "NOT_FOUND"
        assert (await client.call("get_order", {"order_id": "2001"}, BOB)).ok

        elig = (await client.call("check_return_eligibility", {"order_id": "1234"}, ALICE)).data
        assert elig["eligible"] and elig["refundable_amount"] == 350_000
        assert (await client.call("check_return_eligibility", {"order_id": "1235"}, ALICE)).data[
            "reasons"
        ] == ["WINDOW_EXPIRED"]
        assert (await client.call("check_return_eligibility", {"order_id": "1239"}, ALICE)).data[
            "reasons"
        ] == ["CATEGORY_EXCLUDED"]
        assert (await client.call("check_return_eligibility", {"order_id": "1240"}, ALICE)).data[
            "reasons"
        ] == ["ALREADY_REQUESTED"]

        ship = (await client.call("get_shipment_status", {"order_id": "1236"}, ALICE)).data
        assert ship["shipment"]["status"] == "in_transit"
        stock = (await client.call("check_stock", {"sku": "CHG-65W"}, ALICE)).data
        assert stock["items"][0]["status"] == "out_of_stock"
        found = (await client.call("search_products", {"query": "dien thoai"}, ALICE)).data
        assert found["products"][0]["sku"] == "PHN-X100"
