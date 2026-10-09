"""What this shop's assistant can do, from what its data covers and what it has switched off.

A tool is offered only when the data it needs is mapped (`schema_mapping.yaml`) and the shop has
not disabled it (`capabilities.disabled_tools`). A kind of request (refund, return, warranty,
order) is offered only when the shop allows it (`capabilities.request_types`) and, for an order,
when orders can be priced from a catalogue with stock. The assistant is never shown a tool that
would only fail.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

RequestType = Literal["order", "refund", "return", "warranty", "handoff"]
REQUEST_TYPES: tuple[RequestType, ...] = ("order", "refund", "return", "warranty", "handoff")

# The mapped entities each domain tool reads, besides the three every shop has
# (customer, order, order_item).
TOOL_NEEDS: dict[str, tuple[str, ...]] = {
    "get_order": (),
    "list_orders": (),
    "check_return_eligibility": (),
    "check_warranty_eligibility": (),
    "get_shipment_status": ("shipment",),
    "search_products": ("product",),
    "compare_products": ("product",),
    "check_stock": ("product", "inventory"),
    "prepare_order_draft": ("product", "inventory"),
}


@dataclass(frozen=True)
class Offer:
    """What is on offer to the model in this deployment."""

    domain_tools: frozenset[str] | None = None  # None: every data tool
    request_types: tuple[RequestType, ...] = REQUEST_TYPES

    def has_tool(self, name: str) -> bool:
        return self.domain_tools is None or name in self.domain_tools

    def satisfies(self, requires: Iterable[str]) -> bool:
        """`requires` holds tool names and `request:<type>` entries."""
        for need in requires:
            if need.startswith("request:"):
                if need.removeprefix("request:") not in self.request_types:
                    return False
            elif not self.has_tool(need):
                return False
        return True

    def missing_tools(self) -> list[str]:
        return [] if self.domain_tools is None else sorted(set(TOOL_NEEDS) - self.domain_tools)

    def missing_request_types(self) -> list[str]:
        return [t for t in REQUEST_TYPES if t not in self.request_types]


def available_tools(entities: Iterable[str], disabled: Iterable[str] = ()) -> frozenset[str]:
    """The domain tools whose data is mapped and that the shop has not disabled."""
    mapped = set(entities)
    off = set(disabled)
    unknown = off - set(TOOL_NEEDS)
    if unknown:
        raise ValueError(
            f"capabilities.disabled_tools names unknown tools {sorted(unknown)}; "
            f"known: {sorted(TOOL_NEEDS)}"
        )
    return frozenset(
        name for name, needs in TOOL_NEEDS.items() if name not in off and set(needs) <= mapped
    )


def effective_request_types(
    configured: Iterable[str], domain_tools: frozenset[str] | None
) -> tuple[RequestType, ...]:
    """The kinds of request the assistant may prepare. `domain_tools=None` means all exist."""
    wanted = set(configured)
    return tuple(
        t
        for t in REQUEST_TYPES
        if t in wanted
        and (t != "order" or domain_tools is None or "prepare_order_draft" in domain_tools)
    )
