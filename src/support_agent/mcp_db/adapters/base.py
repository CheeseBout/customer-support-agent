"""Adapter contract and value normalisation shared by the SQL and MongoDB adapters.

Adapters return rows keyed by *canonical* field names (SPEC 8.1) with normalised values:
ISO-8601 timestamps carrying a timezone, plain int/float for money, canonical statuses.
All methods are read-only. Ownership is enforced inside the query (`owner` argument), not
by post-filtering, so a row belonging to someone else is never even fetched.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal
from zoneinfo import ZoneInfo

from support_agent.mcp_db.mapping import EntityMapping, SchemaMapping

Row = dict[str, Any]

_DATE_FIELDS = {"created_at", "delivered_at", "updated_at", "eta"}
_NUMBER_FIELDS = {"total", "price", "unit_price", "quantity"}


class AdapterError(RuntimeError):
    """The data source failed (connection, syntax, permission)."""


class AdapterTimeout(AdapterError):
    """The query exceeded the configured timeout."""


@dataclass(frozen=True)
class Issue:
    level: Literal["error", "warning"]
    entity: str
    message: str


def _as_number(value: Any) -> Any:
    if hasattr(value, "to_decimal"):  # bson.Decimal128
        value = value.to_decimal()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            d = Decimal(value.strip())
        except Exception:
            return value
        return int(d) if d == d.to_integral_value() else float(d)
    return value


def _as_bool(value: Any) -> Any:
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "t", "yes", "y"}:
            return True
        if low in {"0", "false", "f", "no", "n", ""}:
            return False
    if isinstance(value, int | float):
        return bool(value)
    return value


def _as_iso(value: Any, naive_tz: ZoneInfo | None) -> Any:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
    else:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=naive_tz or UTC)
    return dt.isoformat()


def normalise_row(
    entity: str, ent: EntityMapping, row: Row, naive_tz: ZoneInfo | None = None
) -> Row:
    out: Row = {}
    for key, value in row.items():
        if value is None:
            out[key] = None
        elif key in _DATE_FIELDS:
            out[key] = _as_iso(value, naive_tz)
        elif key in _NUMBER_FIELDS:
            out[key] = _as_number(value)
        elif key == "active":
            out[key] = _as_bool(value)
        elif key == "attributes" and isinstance(value, str):
            try:
                out[key] = json.loads(value)
            except ValueError:
                out[key] = value
        elif key == "status":
            raw = str(value)
            out[key] = ent.status_map.get(raw, ent.status_map.get(raw.strip(), raw.strip().lower()))
        elif key in {"id", "order_id", "customer_id", "sku"}:
            out[key] = str(value)
        else:
            out[key] = value
    return out


def db_statuses_for(ent: EntityMapping, canonical: str) -> list[str]:
    """Raw DB values that map to a canonical status (reverse of `status_map`)."""
    if not ent.status_map:
        return [canonical]
    values = [raw for raw, canon in ent.status_map.items() if canon == canonical]
    return values or [canonical]


class DataAdapter(ABC):
    def __init__(
        self,
        mapping: SchemaMapping,
        *,
        max_rows: int = 50,
        timeout_seconds: float = 5,
        timezone: str = "Asia/Ho_Chi_Minh",
    ) -> None:
        self.mapping = mapping
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        self.tz = ZoneInfo(timezone)

    # --- orders ----------------------------------------------------------------------
    @abstractmethod
    async def get_order(self, order_id: str, owner: str | None) -> Row | None:
        """The order if it exists and (`owner` is None or belongs to `owner`), else None."""

    @abstractmethod
    async def get_order_items(self, order_id: str) -> list[Row]: ...

    @abstractmethod
    async def list_orders(self, owner: str | None, status: str | None, limit: int) -> list[Row]:
        """Newest first."""

    @abstractmethod
    async def get_shipment(self, order_id: str) -> Row | None: ...

    @abstractmethod
    async def get_return_requests(self, order_id: str) -> list[Row]: ...

    # --- catalogue -------------------------------------------------------------------
    @abstractmethod
    async def get_products(self, skus: list[str]) -> dict[str, Row]: ...

    @abstractmethod
    async def search_products(
        self,
        *,
        category: str | None,
        min_price: float | None,
        max_price: float | None,
        candidate_cap: int,
    ) -> list[Row]:
        """Candidate products (SQL/Mongo-side filters only). Active filtering and text ranking
        happen in the service layer so behaviour is identical across databases."""

    @abstractmethod
    async def get_inventory(self, skus: list[str]) -> dict[str, int]: ...

    # --- lifecycle / introspection ---------------------------------------------------
    @abstractmethod
    async def validate(self) -> list[Issue]:
        """Check the mapping against the live database (SPEC 8.2)."""

    @abstractmethod
    async def close(self) -> None: ...
