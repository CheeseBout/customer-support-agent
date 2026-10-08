"""Domain tool logic over a DataAdapter. Transport-agnostic: the MCP server just wraps it.

Every method takes the authenticated `Principal` explicitly. The owner filter is applied inside
the adapter query, and "not yours" is reported as NOT_FOUND so a caller cannot probe which
order ids exist (SPEC 8.4).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from support_agent.core.principal import Principal
from support_agent.core.results import ToolResult
from support_agent.core.settings import BusinessRules
from support_agent.mcp_db.adapters.base import AdapterError, AdapterTimeout, DataAdapter, Row
from support_agent.rules.eligibility import check_return_eligibility
from support_agent.rules.warranty import check_warranty_eligibility

MAX_ORDER_LIST = 10
MAX_PRODUCTS = 10
CANDIDATE_CAP = 1000
DESCRIPTION_CHARS = 300

_STOPWORDS = {
    "a",
    "an",
    "the",
    "for",
    "me",
    "my",
    "with",
    "and",
    "or",
    "of",
    "to",
    "i",
    "want",
    "need",
    "looking",
    "find",
    "show",
    "please",
    "co",
    "cho",
    "toi",
    "tim",
    "mua",
    "can",
    "muon",
    "nao",
    "nhung",
    "va",
    "hay",
    "gi",
    "khong",
    "la",
    "loai",
    "san",
    "pham",
}


class ProductFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str | None = None
    min_price: float | None = None
    max_price: float | None = None


class OrderLineRequest(BaseModel):
    """One product the customer wants to order. Price is never taken from here."""

    model_config = ConfigDict(extra="forbid")

    sku: str
    qty: int = Field(ge=1)


def match_order_sku(items: list[dict[str, Any]], wanted: str | None) -> str | None:
    """An exact SKU stays as it is. Otherwise a name the customer used ("Boom 5") that matches
    exactly one line of the order is turned into that line's SKU; anything else is left alone,
    so the rules engine reports ITEM_NOT_FOUND."""
    if not wanted:
        return wanted
    text = wanted.strip()
    if any(str(i.get("sku", "")).lower() == text.lower() for i in items):
        return next(str(i["sku"]) for i in items if str(i.get("sku", "")).lower() == text.lower())
    needle = fold(text)
    hits = [
        i
        for i in items
        if needle and needle in fold(f"{i.get('product_name') or ''} {i.get('sku') or ''}")
    ]
    return str(hits[0]["sku"]) if len(hits) == 1 else text


def _with_order_items(data: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    """When the SKU is not on the order, list the order's real lines so the model picks one of
    them (or asks the customer) instead of inventing a SKU."""
    if "ITEM_NOT_FOUND" in data.get("reasons", []):
        data["order_items"] = [
            {"sku": i.get("sku"), "product_name": i.get("product_name")} for i in items
        ]
    return data


def fold(text: str) -> str:
    """Lowercase and strip diacritics so 'điện thoại' matches 'dien thoai'."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def _tokens(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", fold(query))
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


Handler = Callable[[], Awaitable[ToolResult]]


async def _guard(handler: Handler) -> ToolResult:
    """Translate adapter failures into the standard error envelope."""
    try:
        return await handler()
    except AdapterTimeout:
        return ToolResult.failure("TIMEOUT", "The data source did not answer in time.")
    except AdapterError:
        return ToolResult.failure("UPSTREAM_ERROR", "The data source is temporarily unavailable.")


class BusinessService:
    def __init__(
        self,
        adapter: DataAdapter,
        rules: BusinessRules,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.adapter = adapter
        self.rules = rules
        self._clock = clock or (lambda: datetime.now(UTC))

    # --- helpers ---------------------------------------------------------------------
    @staticmethod
    def _owner(principal: Principal) -> str | None:
        """customers see only their own rows; staff are unrestricted."""
        return None if principal.role == "staff" else principal.user_id

    async def _own_order(self, principal: Principal, order_id: str) -> Row | None:
        return await self.adapter.get_order(str(order_id).strip(), self._owner(principal))

    @staticmethod
    def _order_not_found(order_id: str) -> ToolResult:
        return ToolResult.failure("NOT_FOUND", f"Order {order_id!r} was not found.")

    @staticmethod
    def _brief(order: Row) -> Row:
        keys = ("id", "status", "total", "currency", "created_at", "delivered_at")
        return {k: order.get(k) for k in keys if k in order}

    # --- orders ----------------------------------------------------------------------
    async def get_order(self, principal: Principal, order_id: str) -> ToolResult:
        async def run() -> ToolResult:
            order = await self._own_order(principal, order_id)
            if order is None:
                return self._order_not_found(order_id)
            items = await self.adapter.get_order_items(order["id"])
            order = {k: v for k, v in order.items() if k != "customer_id"}
            return ToolResult.success({"order": order, "items": items})

        return await _guard(run)

    async def list_orders(
        self, principal: Principal, status: str | None = None, limit: int = 5
    ) -> ToolResult:
        async def run() -> ToolResult:
            capped = max(1, min(int(limit), MAX_ORDER_LIST))
            rows = await self.adapter.list_orders(self._owner(principal), status, capped)
            return ToolResult.success(
                {"orders": [self._brief(r) for r in rows], "count": len(rows)}
            )

        return await _guard(run)

    async def get_shipment_status(self, principal: Principal, order_id: str) -> ToolResult:
        async def run() -> ToolResult:
            order = await self._own_order(principal, order_id)
            if order is None:
                return self._order_not_found(order_id)
            shipment = await self.adapter.get_shipment(order["id"])
            # Embedded documents (MongoDB) yield an all-null row when no shipment exists.
            if shipment and all(v is None for k, v in shipment.items() if k != "order_id"):
                shipment = None
            data: dict[str, Any] = {"order_id": order["id"], "order_status": order.get("status")}
            if shipment is None:
                data["shipment"] = None
                data["note"] = "No shipment record exists for this order yet."
            else:
                data["shipment"] = {k: v for k, v in shipment.items() if k != "order_id"}
            return ToolResult.success(data)

        return await _guard(run)

    async def check_return_eligibility(
        self, principal: Principal, order_id: str, sku: str | None = None
    ) -> ToolResult:
        async def run() -> ToolResult:
            order = await self._own_order(principal, order_id)
            if order is None:
                return self._order_not_found(order_id)
            items = await self.adapter.get_order_items(order["id"])
            products = await self.adapter.get_products([str(i["sku"]) for i in items])
            requests = await self.adapter.get_return_requests(order["id"])
            result = check_return_eligibility(
                order=order,
                items=items,
                products=products,
                existing_requests=requests,
                requester_id=principal.user_id,
                requester_is_staff=principal.role == "staff",
                rule=self.rules.return_,
                now=self._clock(),
                tz=self.adapter.tz,
                sku=match_order_sku(items, sku),
            )
            return ToolResult.success(
                _with_order_items(
                    {"order_id": order["id"], **result.model_dump(mode="json")}, items
                )
            )

        return await _guard(run)

    # --- catalogue -------------------------------------------------------------------
    async def search_products(
        self,
        principal: Principal,
        query: str,
        filters: ProductFilters | None = None,
        limit: int = 5,
    ) -> ToolResult:
        async def run() -> ToolResult:
            tokens = _tokens(query)
            if not tokens:
                return ToolResult.failure("INVALID_ARGUMENT", "The search query is empty.")
            f = filters or ProductFilters()
            candidates = await self.adapter.search_products(
                category=f.category,
                min_price=f.min_price,
                max_price=f.max_price,
                candidate_cap=CANDIDATE_CAP,
            )
            scored: list[tuple[int, Row]] = []
            for p in candidates:
                if p.get("active") is False:
                    continue
                haystack = fold(
                    " ".join(str(p.get(k) or "") for k in ("name", "category", "description"))
                    + " "
                    + json.dumps(p.get("attributes") or {}, ensure_ascii=False)
                )
                hits = sum(1 for t in tokens if t in haystack)
                if hits:
                    scored.append((hits, p))
            scored.sort(key=lambda x: (-x[0], str(x[1].get("name"))))
            capped = max(1, min(int(limit), MAX_PRODUCTS))
            products = [self._product_view(p) for _, p in scored[:capped]]
            return ToolResult.success({"products": products, "count": len(products)})

        return await _guard(run)

    @staticmethod
    def _product_view(p: Row) -> Row:
        desc = str(p.get("description") or "")
        return {
            "sku": p.get("sku"),
            "name": p.get("name"),
            "price": p.get("price"),
            "category": p.get("category"),
            "description": desc[:DESCRIPTION_CHARS]
            + ("…" if len(desc) > DESCRIPTION_CHARS else ""),
            "attributes": p.get("attributes"),
        }

    async def check_stock(
        self, principal: Principal, sku: str | None = None, query: str | None = None
    ) -> ToolResult:
        async def run() -> ToolResult:
            if not sku and not query:
                return ToolResult.failure("INVALID_ARGUMENT", "Provide a `sku` or a `query`.")
            if sku:
                products = await self.adapter.get_products([sku.strip()])
                products = {k: v for k, v in products.items() if v.get("active") is not False}
                if not products:
                    return ToolResult.failure("NOT_FOUND", f"Product {sku!r} was not found.")
            else:
                found = await self.search_products(principal, query or "", limit=5)
                if not found.ok:
                    return found
                products = {p["sku"]: p for p in (found.data or {}).get("products", [])}
                if not products:
                    return ToolResult.failure("NOT_FOUND", f"No product matches {query!r}.")
            stock = await self.adapter.get_inventory(list(products))
            show_qty = self.rules.inventory.show_exact_quantity or principal.role == "staff"
            out = []
            for key, p in products.items():
                qty = stock.get(key, 0)
                entry: Row = {
                    "sku": key,
                    "name": p.get("name"),
                    "status": self._stock_status(qty),
                }
                if show_qty:
                    entry["quantity"] = qty
                out.append(entry)
            return ToolResult.success({"items": out})

        return await _guard(run)

    async def check_warranty_eligibility(
        self, principal: Principal, order_id: str, sku: str
    ) -> ToolResult:
        async def run() -> ToolResult:
            order = await self._own_order(principal, order_id)
            if order is None:
                return self._order_not_found(order_id)
            items = await self.adapter.get_order_items(order["id"])
            products = await self.adapter.get_products([str(i["sku"]) for i in items])
            result = check_warranty_eligibility(
                order=order,
                items=items,
                products=products,
                sku=match_order_sku(items, sku) or sku.strip(),
                requester_id=principal.user_id,
                requester_is_staff=principal.role == "staff",
                rule=self.rules.warranty,
                now=self._clock(),
                tz=self.adapter.tz,
            )
            return ToolResult.success(
                _with_order_items(
                    {"order_id": order["id"], **result.model_dump(mode="json")}, items
                )
            )

        return await _guard(run)

    async def compare_products(self, principal: Principal, skus: list[str]) -> ToolResult:
        """Side-by-side specs of 2 to 4 products, as rows the model can turn into a table."""

        async def run() -> ToolResult:
            unique = list(dict.fromkeys(s.strip() for s in skus if s and s.strip()))
            if not 2 <= len(unique) <= 4:
                return ToolResult.failure("INVALID_ARGUMENT", "Give between 2 and 4 distinct SKUs.")
            found = await self.adapter.get_products(unique)
            active = {k: v for k, v in found.items() if v.get("active") is not False}
            missing = [s for s in unique if s not in active]
            if len(active) < 2:
                return ToolResult.failure(
                    "NOT_FOUND", f"Fewer than two of those products exist. Not found: {missing}."
                )
            columns = [s for s in unique if s in active]
            stock = await self.adapter.get_inventory(columns)
            show_qty = self.rules.inventory.show_exact_quantity or principal.role == "staff"

            def attributes(sku: str) -> dict[str, Any]:
                raw = active[sku].get("attributes")
                return raw if isinstance(raw, dict) else {}

            rows: list[Row] = [
                {"attribute": "name", "values": {s: active[s].get("name") for s in columns}},
                {
                    "attribute": "category",
                    "values": {s: active[s].get("category") for s in columns},
                },
                {"attribute": "price", "values": {s: active[s].get("price") for s in columns}},
                {
                    "attribute": "availability",
                    "values": {
                        s: self._stock_status(stock.get(s, 0))
                        + (f" ({stock.get(s, 0)})" if show_qty else "")
                        for s in columns
                    },
                },
            ]
            for key in sorted({k for s in columns for k in attributes(s)}):
                rows.append(
                    {"attribute": key, "values": {s: attributes(s).get(key) for s in columns}}
                )
            return ToolResult.success({"skus": columns, "not_found": missing, "rows": rows})

        return await _guard(run)

    async def prepare_order_draft(
        self,
        principal: Principal,
        items: list[OrderLineRequest],
        shipping_address: str,
        payment_method: str,
    ) -> ToolResult:
        """Validate an order and price it from the database. Writes nothing (SPEC 8.4).

        Prices and totals always come from the product table: an amount typed by the customer
        or invented by the model never reaches the draft.
        """

        async def run() -> ToolResult:
            rule = self.rules.order
            wanted: dict[str, int] = {}
            for line in items:
                wanted[line.sku.strip()] = wanted.get(line.sku.strip(), 0) + line.qty
            if not wanted:
                return ToolResult.failure("INVALID_ARGUMENT", "Add at least one product.")
            if len(wanted) > rule.max_lines:
                return ToolResult.failure(
                    "LIMIT_EXCEEDED",
                    f"An order can have at most {rule.max_lines} different products.",
                )
            too_many = sorted(s for s, q in wanted.items() if q > rule.max_quantity_per_line)
            if too_many:
                return ToolResult.failure(
                    "LIMIT_EXCEEDED",
                    f"At most {rule.max_quantity_per_line} of one product per order: {too_many}.",
                )
            address = " ".join(shipping_address.split())
            if len(address) < 5:
                return ToolResult.failure(
                    "INVALID_ARGUMENT", "A full shipping address is required."
                )
            method = payment_method.strip().lower().replace(" ", "_").replace("-", "_")
            if method not in rule.payment_methods:
                return ToolResult.failure(
                    "NOT_ELIGIBLE", f"Payment method must be one of {rule.payment_methods}."
                )

            found = await self.adapter.get_products(list(wanted))
            usable = {k: v for k, v in found.items() if v.get("active") is not False}
            unknown = sorted(set(wanted) - set(usable))
            if unknown:
                return ToolResult.failure("NOT_FOUND", f"Not available for sale: {unknown}.")
            stock = await self.adapter.get_inventory(list(wanted))
            short = sorted(s for s, q in wanted.items() if stock.get(s, 0) < q)
            if short:
                detail = (
                    {s: stock.get(s, 0) for s in short}
                    if self.rules.inventory.show_exact_quantity or principal.role == "staff"
                    else short
                )
                return ToolResult.failure("OUT_OF_STOCK", f"Not enough stock for: {detail}.")

            lines: list[dict[str, Any]] = [
                {
                    "sku": s,
                    "name": usable[s].get("name"),
                    "qty": q,
                    "unit_price": usable[s].get("price"),
                }
                for s, q in wanted.items()
            ]
            total = sum((ln["unit_price"] or 0) * ln["qty"] for ln in lines)
            if method == "cod" and rule.cod_max_total is not None and total > rule.cod_max_total:
                return ToolResult.failure(
                    "NOT_ELIGIBLE",
                    f"Cash on delivery is limited to {rule.cod_max_total} {rule.currency}; "
                    "choose another payment method.",
                )
            return ToolResult.success(
                {
                    "items": lines,
                    "shipping_address": address,
                    "payment_method": method,
                    "total": total,
                    "currency": rule.currency,
                }
            )

        return await _guard(run)

    def _stock_status(self, qty: int) -> str:
        if qty <= 0:
            return "out_of_stock"
        if qty <= self.rules.inventory.low_stock_threshold:
            return "low_stock"
        return "in_stock"
