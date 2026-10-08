"""MCP server exposing the read-only domain tools (SPEC 8.3).

Identity never travels in tool arguments. The client attaches a signed principal to the
request `_meta`; every tool verifies it before touching data and refuses any argument that
is not part of the tool's declared signature.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from support_agent.core.principal import (
    InvalidPrincipal,
    Principal,
    hash_user_id,
    verify_principal,
)
from support_agent.core.results import ToolResult
from support_agent.mcp_db.service import BusinessService, OrderLineRequest, ProductFilters
from support_agent.security.pii import mask_text

PRINCIPAL_META_KEY = "support_agent/principal"
# Arguments that try to carry an identity are a prompt-injection signature: refuse loudly.
IDENTITY_ARGS = {"customer_id", "user_id", "principal", "role", "owner", "customer"}

log = logging.getLogger(__name__)
audit = logging.getLogger("support_agent.audit")

Call = Callable[[Principal], Awaitable[ToolResult]]


def _arguments(ctx: Context) -> dict[str, Any]:
    params = ctx.request_context.params
    raw = (
        params.get("arguments") if isinstance(params, dict) else getattr(params, "arguments", None)
    )
    return raw if isinstance(raw, dict) else {}


def build_server(service: BusinessService, secret: bytes) -> MCPServer:
    server = MCPServer(
        "support-agent-db",
        instructions="Read-only domain tools over the shop database. Identity is implicit.",
    )

    async def run(ctx: Context, tool: str, allowed: set[str], call: Call) -> dict[str, Any]:
        started = time.perf_counter()
        args = _arguments(ctx)
        principal: Principal | None = None
        try:
            meta = ctx.request_context.meta or {}
            principal = verify_principal(meta.get(PRINCIPAL_META_KEY), secret)
        except InvalidPrincipal as exc:
            result = ToolResult.failure("FORBIDDEN", "Missing or invalid caller identity.")
            log.warning("tool %s rejected: %s", tool, exc)
        else:
            extra = set(args) - allowed
            if extra & IDENTITY_ARGS:
                result = ToolResult.failure(
                    "FORBIDDEN", "Identity cannot be supplied as a tool argument."
                )
                log.warning(
                    "tool %s: identity argument(s) refused: %s", tool, sorted(extra & IDENTITY_ARGS)
                )
            elif extra:
                result = ToolResult.failure(
                    "INVALID_ARGUMENT", f"Unexpected argument(s): {sorted(extra)}"
                )
            else:
                result = await call(principal)

        audit.info(
            "tool_call",
            extra={
                "tool": tool,
                "caller": hash_user_id(principal.user_id) if principal else None,
                "tool_args": mask_text(json.dumps(args, ensure_ascii=False, default=str)),
                "result_code": "OK"
                if result.ok
                else (result.error.code if result.error else "ERR"),
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return result.to_wire()

    @server.tool(
        name="get_order",
        description="Get one of the customer's own orders (status, totals, dates, items) "
        "by order id.",
    )
    async def get_order(order_id: str, ctx: Context) -> dict[str, Any]:
        return await run(ctx, "get_order", {"order_id"}, lambda p: service.get_order(p, order_id))

    @server.tool(
        name="list_orders",
        description="List the customer's own recent orders, newest first. Optional status filter "
        "(processing|shipping|delivered|cancelled) and limit (max 10).",
    )
    async def list_orders(
        ctx: Context, status: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        return await run(
            ctx, "list_orders", {"status", "limit"}, lambda p: service.list_orders(p, status, limit)
        )

    @server.tool(
        name="get_shipment_status",
        description="Get carrier, tracking code, status and ETA for one of the customer's "
        "own orders.",
    )
    async def get_shipment_status(order_id: str, ctx: Context) -> dict[str, Any]:
        return await run(
            ctx,
            "get_shipment_status",
            {"order_id"},
            lambda p: service.get_shipment_status(p, order_id),
        )

    @server.tool(
        name="check_stock",
        description="Check stock status (in_stock|low_stock|out_of_stock) by exact `sku` or by "
        "product name `query`.",
    )
    async def check_stock(
        ctx: Context, sku: str | None = None, query: str | None = None
    ) -> dict[str, Any]:
        return await run(
            ctx, "check_stock", {"sku", "query"}, lambda p: service.check_stock(p, sku, query)
        )

    @server.tool(
        name="search_products",
        description="Search active products by keywords, optional filters {category, min_price, "
        "max_price}, limit max 10.",
    )
    async def search_products(
        query: str, ctx: Context, filters: ProductFilters | None = None, limit: int = 5
    ) -> dict[str, Any]:
        return await run(
            ctx,
            "search_products",
            {"query", "filters", "limit"},
            lambda p: service.search_products(p, query, filters, limit),
        )

    @server.tool(
        name="check_return_eligibility",
        description="Run the shop's return rules for one of the customer's own orders (optionally "
        "one `sku`). Returns eligible, reason codes, deadline and refundable amount.",
    )
    async def check_return_eligibility(
        order_id: str, ctx: Context, sku: str | None = None
    ) -> dict[str, Any]:
        return await run(
            ctx,
            "check_return_eligibility",
            {"order_id", "sku"},
            lambda p: service.check_return_eligibility(p, order_id, sku),
        )

    @server.tool(
        name="check_warranty_eligibility",
        description="Run the shop's warranty rules for one `sku` of one of the customer's own "
        "orders. Returns eligible, reason codes, warranty period and expiry date.",
    )
    async def check_warranty_eligibility(order_id: str, sku: str, ctx: Context) -> dict[str, Any]:
        return await run(
            ctx,
            "check_warranty_eligibility",
            {"order_id", "sku"},
            lambda p: service.check_warranty_eligibility(p, order_id, sku),
        )

    @server.tool(
        name="compare_products",
        description="Compare 2 to 4 products side by side by SKU: price, category, availability "
        "and every specification, as rows to present as a table.",
    )
    async def compare_products(skus: list[str], ctx: Context) -> dict[str, Any]:
        return await run(
            ctx, "compare_products", {"skus"}, lambda p: service.compare_products(p, skus)
        )

    @server.tool(
        name="prepare_order_draft",
        description="Check stock and price an order from the database (items: sku + qty, "
        "shipping address, payment method). Writes nothing; use it to quote or before proposing.",
    )
    async def prepare_order_draft(
        items: list[OrderLineRequest], shipping_address: str, payment_method: str, ctx: Context
    ) -> dict[str, Any]:
        return await run(
            ctx,
            "prepare_order_draft",
            {"items", "shipping_address", "payment_method"},
            lambda p: service.prepare_order_draft(p, items, shipping_address, payment_method),
        )

    return server


def main() -> None:  # pragma: no cover - process entry point
    from support_agent.core.logging import configure_logging
    from support_agent.core.settings import get_settings
    from support_agent.mcp_db.factory import create_adapter
    from support_agent.mcp_db.mapping import load_mapping

    settings = get_settings()
    configure_logging(settings.log_level)
    if settings.mcp_principal_secret is None:
        raise SystemExit("MCP_PRINCIPAL_SECRET must be set for the MCP server process")
    mapping = load_mapping(settings.app.mapping.path)
    adapter = create_adapter(settings, mapping)
    service = BusinessService(adapter, settings.app.business_rules)
    secret = settings.mcp_principal_secret.get_secret_value().encode()
    build_server(service, secret).run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
