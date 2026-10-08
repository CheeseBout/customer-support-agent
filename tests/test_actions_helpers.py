"""Small builders shared by the after-sales tests."""

from __future__ import annotations

from support_agent.mcp_db.service import OrderLineRequest


def lines(*pairs: tuple[str, int]) -> list[OrderLineRequest]:
    """`lines(("EAR-BT20", 2))` -> the order-line objects the service takes."""
    return [OrderLineRequest.model_construct(sku=sku, qty=qty) for sku, qty in pairs]
