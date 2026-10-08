"""LangChain tool wrappers. The schemas the LLM sees contain no identity parameter at all."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from support_agent.core.principal import Principal
from support_agent.core.results import ToolResult
from support_agent.mcp_db.service import ProductFilters
from support_agent.tools.client import DomainToolClient


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # `customer_id` etc. fail validation here


class GetOrderArgs(ToolArgs):
    order_id: str = Field(description="Order id, e.g. 1234 (no leading #)")


class ListOrdersArgs(ToolArgs):
    status: str | None = Field(default=None, description="processing|shipping|delivered|cancelled")
    limit: int = Field(default=5, ge=1, le=10)


class ShipmentArgs(ToolArgs):
    order_id: str


class CheckStockArgs(ToolArgs):
    sku: str | None = Field(default=None, description="Exact SKU")
    query: str | None = Field(default=None, description="Product name keywords")


class SearchProductsArgs(ToolArgs):
    query: str
    filters: ProductFilters | None = None
    limit: int = Field(default=5, ge=1, le=10)


class ReturnEligibilityArgs(ToolArgs):
    order_id: str
    sku: str | None = None


class WarrantyArgs(ToolArgs):
    order_id: str
    sku: str = Field(description="The product (SKU) the warranty claim is about")


class CompareArgs(ToolArgs):
    skus: list[str] = Field(min_length=2, max_length=4, description="2 to 4 SKUs to compare")


class OrderItemArgs(ToolArgs):
    sku: str
    qty: int = Field(ge=1, le=100)


class PrepareOrderArgs(ToolArgs):
    items: list[OrderItemArgs] = Field(min_length=1, max_length=20)
    shipping_address: str = Field(description="Full delivery address")
    payment_method: str = Field(description="cod, bank_transfer, card, momo, zalopay or vnpay")


TOOL_SPECS: list[tuple[str, str, type[BaseModel]]] = [
    (
        "get_order",
        "Get one of the customer's own orders (status, totals, dates, items).",
        GetOrderArgs,
    ),
    ("list_orders", "List the customer's own recent orders, newest first.", ListOrdersArgs),
    ("get_shipment_status", "Carrier, tracking code, status and ETA of an order.", ShipmentArgs),
    ("check_stock", "Stock status by exact SKU or product-name query.", CheckStockArgs),
    (
        "search_products",
        "Search active products by keywords and optional filters.",
        SearchProductsArgs,
    ),
    (
        "check_return_eligibility",
        "Run the shop's return rules for an order (optionally one SKU).",
        ReturnEligibilityArgs,
    ),
    (
        "check_warranty_eligibility",
        "Run the shop's warranty rules for one SKU of an order.",
        WarrantyArgs,
    ),
    (
        "compare_products",
        "Compare 2 to 4 products side by side: price, availability and every specification.",
        CompareArgs,
    ),
    (
        "prepare_order_draft",
        "Check stock and price an order from the database. Writes nothing.",
        PrepareOrderArgs,
    ),
]


def build_domain_tools(client: DomainToolClient, principal: Principal) -> list[BaseTool]:
    """Tools bound to `principal`. The principal comes from the session, never from the LLM."""

    def make(name: str, description: str, schema: type[BaseModel]) -> BaseTool:
        async def run(**kwargs: Any) -> str:
            args = {k: v for k, v in kwargs.items() if v is not None}
            result = await client.call(name, args, principal)
            return json.dumps(result.to_wire(), ensure_ascii=False)

        return StructuredTool.from_function(
            coroutine=run, name=name, description=description, args_schema=schema
        )

    return [make(n, d, s) for n, d, s in TOOL_SPECS]


DOMAIN_TOOL_NAMES = frozenset(name for name, _, _ in TOOL_SPECS)
ARGS_SCHEMAS: dict[str, type[BaseModel]] = {name: schema for name, _, schema in TOOL_SPECS}


async def _never_called(**_: Any) -> str:
    raise RuntimeError("schema-only tool: the agent's tool node executes calls")


def schema_tool(name: str, description: str, schema: type[BaseModel]) -> BaseTool:
    """A tool that only describes itself to the model.

    Execution happens in the agent's tool node, which takes the identity from the run state,
    so no principal is ever captured in (or visible to) the tool definition.
    """
    return StructuredTool.from_function(
        coroutine=_never_called, name=name, description=description, args_schema=schema
    )


def domain_schema_tools() -> list[BaseTool]:
    return [schema_tool(n, d, s) for n, d, s in TOOL_SPECS]


def parse_tool_output(content: str) -> ToolResult:
    return ToolResult.model_validate(json.loads(content))
