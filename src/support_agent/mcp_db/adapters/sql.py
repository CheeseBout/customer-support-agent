"""SQLAlchemy (async) adapter for PostgreSQL, MySQL and SQLite.

Queries are built with SQLAlchemy Core from the validated mapping, so every value is a bound
parameter and identifiers come only from the admin-authored mapping file.
"""

from __future__ import annotations

import asyncio
import functools
import operator
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from support_agent.mcp_db.adapters.base import (
    AdapterError,
    AdapterTimeout,
    DataAdapter,
    Issue,
    Row,
    db_statuses_for,
    normalise_row,
)
from support_agent.mcp_db.mapping import (
    CANONICAL_ORDER_STATUSES,
    Col,
    EntityMapping,
    Expr,
    Lit,
    SchemaMapping,
    columns_in,
    parse_expression,
)

_NUMERIC_FIELDS = {
    "total",
    "price",
    "unit_price",
    "quantity",
    "subtotal",
    "shipping_fee",
    "discount",
    "tax",
    "refunded_amount",
}
_DATE_FIELDS = {"created_at", "delivered_at", "updated_at", "eta"}


def compile_expr(expr: Expr) -> sa.ColumnElement[Any]:
    if isinstance(expr, Col):
        return sa.column(expr.path)
    if isinstance(expr, Lit):
        return sa.literal(expr.value, sa.String() if isinstance(expr.value, str) else None)
    parts = [compile_expr(a) for a in expr.args]
    if expr.name == "COALESCE":
        return sa.func.coalesce(*parts)
    # CONCAT: cast every part to text so `+` renders as `||` (or concat() on MySQL).
    return functools.reduce(operator.add, [sa.cast(p, sa.String()) for p in parts])


def _on_connect(engine: AsyncEngine, statement: str) -> None:
    @event.listens_for(engine.sync_engine, "connect")
    def _run(dbapi_conn: Any, _record: Any) -> None:
        cur = dbapi_conn.cursor()
        cur.execute(statement)
        cur.close()


def _as_text(expr: sa.ColumnElement[Any]) -> sa.ColumnElement[str]:
    """Compare ids as text so integer ids and string ids both work with string inputs."""
    return sa.cast(expr, sa.String())


class SqlAdapter(DataAdapter):
    def __init__(self, engine: AsyncEngine, mapping: SchemaMapping, **kwargs: Any) -> None:
        super().__init__(mapping, **kwargs)
        self.engine = engine
        self._exprs: dict[str, dict[str, sa.ColumnElement[Any]]] = {}
        self._tables: dict[str, sa.TableClause] = {}
        for name, ent in mapping.entities.items():
            self._tables[name] = self._table(ent)
            self._exprs[name] = {
                f: compile_expr(parse_expression(t)) for f, t in ent.fields.items()
            }

    # --- construction ----------------------------------------------------------------
    @staticmethod
    def _table(ent: EntityMapping) -> sa.TableClause:
        assert ent.table
        schema, _, name = ent.table.rpartition(".")
        return sa.table(name, schema=schema or None)

    @classmethod
    def from_url(cls, url: str, mapping: SchemaMapping, **kwargs: Any) -> SqlAdapter:
        backend = make_url(url).get_backend_name()
        if backend == "sqlite":
            engine = create_async_engine(url)
        else:
            engine = create_async_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=2)

        # Defence in depth: the DB account should already be read-only (SPEC NFR-002).
        if backend == "postgresql":
            engine = engine.execution_options(postgresql_readonly=True)
        elif backend in ("mysql", "mariadb"):
            _on_connect(engine, "SET SESSION TRANSACTION READ ONLY")
        elif backend == "sqlite":
            _on_connect(engine, "PRAGMA query_only = ON")
        return cls(engine, mapping, **kwargs)

    # --- query plumbing --------------------------------------------------------------
    def _select(self, entity: str, fields: Sequence[str] | None = None) -> sa.Select[Any]:
        exprs = self._exprs[entity]
        names = [f for f in (fields or exprs) if f in exprs]
        return sa.select(*(exprs[f].label(f) for f in names)).select_from(self._tables[entity])

    def _where(self, entity: str, field: str) -> sa.ColumnElement[Any]:
        return self._exprs[entity][field]

    async def _fetch(self, stmt: sa.Select[Any], limit: int | None = None) -> list[dict[str, Any]]:
        """Run `stmt` with a row cap. `limit=None` means the configured `max_rows`."""
        stmt = stmt.limit(self.max_rows if limit is None else limit)

        async def run() -> list[dict[str, Any]]:
            async with self.engine.connect() as conn:
                result = await conn.execute(stmt)
                return [dict(r) for r in result.mappings().all()]

        try:
            return await asyncio.wait_for(run(), timeout=self.timeout_seconds)
        except TimeoutError as exc:
            raise AdapterTimeout(f"query exceeded {self.timeout_seconds}s") from exc
        except SQLAlchemyError as exc:
            # Never echo the driver message: it can contain SQL text and bound values.
            raise AdapterError(f"database error: {type(exc).__name__}") from exc

    def _norm(self, entity: str, rows: list[dict[str, Any]]) -> list[Row]:
        ent = self.mapping.entity(entity)
        return [normalise_row(entity, ent, r, self.tz) for r in rows]

    # --- orders ----------------------------------------------------------------------
    async def get_order(self, order_id: str, owner: str | None) -> Row | None:
        stmt = self._select("order").where(_as_text(self._where("order", "id")) == str(order_id))
        if owner is not None:
            stmt = stmt.where(_as_text(self._where("order", "customer_id")) == str(owner))
        rows = self._norm("order", await self._fetch(stmt))
        return rows[0] if rows else None

    async def get_order_items(self, order_id: str) -> list[Row]:
        stmt = self._select("order_item").where(
            _as_text(self._where("order_item", "order_id")) == str(order_id)
        )
        return self._norm("order_item", await self._fetch(stmt))

    async def list_orders(self, owner: str | None, status: str | None, limit: int) -> list[Row]:
        ent = self.mapping.entity("order")
        stmt = self._select("order")
        if owner is not None:
            stmt = stmt.where(_as_text(self._where("order", "customer_id")) == str(owner))
        if status:
            raw = db_statuses_for(ent, status)
            stmt = stmt.where(_as_text(self._where("order", "status")).in_([str(s) for s in raw]))
        stmt = stmt.order_by(self._where("order", "created_at").desc())
        return self._norm("order", await self._fetch(stmt, min(limit, self.max_rows)))

    async def get_shipments(self, order_id: str) -> list[Row]:
        if not self.mapping.has("shipment"):
            return []
        stmt = self._select("shipment").where(
            _as_text(self._where("shipment", "order_id")) == str(order_id)
        )
        if "updated_at" in self._exprs["shipment"]:
            stmt = stmt.order_by(self._where("shipment", "updated_at").desc())
        return self._norm("shipment", await self._fetch(stmt))

    async def get_return_requests(self, order_id: str) -> list[Row]:
        if not self.mapping.has("return_request"):
            return []
        stmt = self._select("return_request").where(
            _as_text(self._where("return_request", "order_id")) == str(order_id)
        )
        return self._norm("return_request", await self._fetch(stmt))

    # --- catalogue -------------------------------------------------------------------
    async def get_products(self, skus: list[str]) -> dict[str, Row]:
        if not skus or not self.mapping.has("product"):
            return {}
        stmt = self._select("product").where(
            _as_text(self._where("product", "sku")).in_([str(s) for s in skus])
        )
        return {r["sku"]: r for r in self._norm("product", await self._fetch(stmt))}

    async def search_products(
        self,
        *,
        category: str | None,
        min_price: float | None,
        max_price: float | None,
        candidate_cap: int,
    ) -> list[Row]:
        if not self.mapping.has("product"):
            return []
        stmt = self._select("product")
        exprs = self._exprs["product"]
        if category and "category" in exprs:
            stmt = stmt.where(sa.func.lower(_as_text(exprs["category"])) == category.lower())
        if min_price is not None:
            stmt = stmt.where(exprs["price"] >= min_price)
        if max_price is not None:
            stmt = stmt.where(exprs["price"] <= max_price)
        rows = await self._fetch(stmt, min(candidate_cap, 1000))
        return self._norm("product", rows)

    async def get_inventory(self, skus: list[str]) -> dict[str, int]:
        if not skus or not self.mapping.has("inventory"):
            return {}
        stmt = self._select("inventory").where(
            _as_text(self._where("inventory", "sku")).in_([str(s) for s in skus])
        )
        out: dict[str, int] = {}
        for r in self._norm("inventory", await self._fetch(stmt)):
            qty = r.get("quantity")
            out[r["sku"]] = out.get(r["sku"], 0) + (int(qty) if isinstance(qty, int | float) else 0)
        return out

    # --- validation ------------------------------------------------------------------
    async def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        try:
            async with self.engine.connect() as conn:
                issues += await conn.run_sync(self._validate_structure)
                for name in ("order", "shipment", "return_request"):
                    if self.mapping.has(name):
                        issues += await self._validate_statuses(conn, name)
        except SQLAlchemyError as exc:
            issues.append(Issue("error", "-", f"cannot inspect database: {type(exc).__name__}"))
        except OSError as exc:
            issues.append(Issue("error", "-", f"cannot connect to database: {exc}"))
        return issues

    def _validate_structure(self, sync_conn: Any) -> list[Issue]:
        inspector = sa.inspect(sync_conn)
        issues: list[Issue] = []
        for name, ent in self.mapping.entities.items():
            schema, _, table = (ent.table or "").rpartition(".")
            schema_arg = schema or None
            exists = inspector.has_table(table, schema=schema_arg) or (
                table in inspector.get_view_names(schema=schema_arg)
            )
            if not exists:
                issues.append(Issue("error", name, f"table or view {ent.table!r} does not exist"))
                continue
            cols = {c["name"]: c for c in inspector.get_columns(table, schema=schema_arg)}
            for field, text in ent.fields.items():
                for col in columns_in(parse_expression(text)):
                    if col not in cols:
                        issues.append(
                            Issue(
                                "error", name, f"{field}: column {col!r} not found in {ent.table}"
                            )
                        )
                        continue
                    issues += self._type_issue(name, field, col, cols[col]["type"])
        return issues

    @staticmethod
    def _type_issue(entity: str, field: str, col: str, col_type: Any) -> list[Issue]:
        try:
            py_type = col_type.python_type
        except NotImplementedError:
            return []  # exotic/unknown type: cannot judge
        if field in _NUMERIC_FIELDS and not issubclass(py_type, int | float | Decimal):
            return [
                Issue("error", entity, f"{field}: column {col!r} is {col_type}, expected a number")
            ]
        if field in _DATE_FIELDS and not issubclass(py_type, datetime | date):
            return [
                Issue(
                    "warning",
                    entity,
                    f"{field}: column {col!r} is {col_type}, expected a date/time",
                )
            ]
        return []

    async def _validate_statuses(self, conn: Any, entity: str) -> list[Issue]:
        ent = self.mapping.entity(entity)
        expr = self._exprs[entity]["status"]
        stmt = (
            sa.select(_as_text(expr).label("s"))
            .select_from(self._tables[entity])
            .distinct()
            .limit(200)
        )
        try:
            values = {str(r[0]) for r in (await conn.execute(stmt)).all() if r[0] is not None}
        except SQLAlchemyError:
            return []  # structural validation already reported the cause
        known = set(ent.status_map) if ent.status_map else CANONICAL_ORDER_STATUSES
        unmapped = sorted(values - known)
        if unmapped and entity == "order":
            return [Issue("error", entity, f"status values not covered by status_map: {unmapped}")]
        if unmapped:
            return [Issue("warning", entity, f"unmapped status values (kept as-is): {unmapped}")]
        return []

    async def close(self) -> None:
        await self.engine.dispose()
