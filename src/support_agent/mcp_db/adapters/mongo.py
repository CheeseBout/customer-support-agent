"""MongoDB (Motor) adapter. Separate from the SQL adapter by design (SPEC 8.2, PLAN risks).

Each entity maps to a collection and dotted field paths. Embedded arrays (e.g. order lines in
`orders.items`) are exposed with `unwind: items`; the adapter then builds
`$match -> $unwind -> $project` pipelines. Reads only; no write operation exists here.
"""

from __future__ import annotations

import re
from typing import Any, Literal

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
    Expr,
    Lit,
    SchemaMapping,
    columns_in,
    parse_expression,
)


def compile_expr(expr: Expr) -> Any:
    """Compile a mapping expression to a MongoDB aggregation expression."""
    if isinstance(expr, Col):
        return f"${expr.path}"
    if isinstance(expr, Lit):
        return {"$literal": expr.value}
    parts = [compile_expr(a) for a in expr.args]
    if expr.name == "COALESCE":
        out = parts[-1]
        for p in reversed(parts[:-1]):
            out = {"$ifNull": [p, out]}
        return out
    return {"$concat": [{"$toString": p} for p in parts]}


def _id_variants(value: str) -> list[Any]:
    """An id typed by a user can be stored as a string or a number; match both."""
    variants: list[Any] = [value]
    if re.fullmatch(r"-?\d+", value):
        variants.append(int(value))
    return variants


class MongoAdapter(DataAdapter):
    def __init__(self, db: Any, mapping: SchemaMapping, client: Any = None, **kwargs: Any) -> None:
        super().__init__(mapping, **kwargs)
        self.db = db
        self._client = client
        self._parsed = {
            name: {f: parse_expression(t) for f, t in ent.fields.items()}
            for name, ent in mapping.entities.items()
        }

    @classmethod
    def from_url(cls, url: str, mapping: SchemaMapping, **kwargs: Any) -> MongoAdapter:
        from motor.motor_asyncio import AsyncIOMotorClient

        timeout_ms = int(kwargs.get("timeout_seconds", 5) * 1000)
        client: Any = AsyncIOMotorClient(url, tz_aware=True, serverSelectionTimeoutMS=timeout_ms)
        db = client.get_default_database()  # the database name comes from the URL path
        return cls(db, mapping, client=client, **kwargs)

    # --- pipeline construction -------------------------------------------------------
    def _match_stage(self, entity: str, field: str, values: list[Any]) -> dict[str, Any]:
        expr = self._parsed[entity][field]
        if isinstance(expr, Col):
            return {expr.path: {"$in": values}}
        return {"$expr": {"$in": [compile_expr(expr), values]}}

    def _pipeline(
        self,
        entity: str,
        matches: list[dict[str, Any]],
        *,
        sort: tuple[str, int] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        ent = self.mapping.entity(entity)
        pre: list[dict[str, Any]] = []
        post: list[dict[str, Any]] = []
        for m in matches:
            touches_array = ent.unwind and any(
                k == ent.unwind or k.startswith(f"{ent.unwind}.")
                for k in m
                if not k.startswith("$")
            )
            (post if touches_array else pre).append(m)
        stages: list[dict[str, Any]] = [{"$match": m} for m in pre]
        if ent.unwind:
            stages.append({"$unwind": f"${ent.unwind}"})
        stages += [{"$match": m} for m in post]
        stages.append(
            {
                "$project": {
                    "_id": 0,
                    **{f: compile_expr(e) for f, e in self._parsed[entity].items()},
                }
            }
        )
        if sort:
            stages.append({"$sort": {sort[0]: sort[1]}})
        stages.append({"$limit": self.max_rows if limit is None else limit})
        return stages

    async def _run(self, entity: str, pipeline: list[dict[str, Any]]) -> list[Row]:
        ent = self.mapping.entity(entity)
        limit = next((s["$limit"] for s in pipeline if "$limit" in s), self.max_rows)
        try:
            cursor = self.db[ent.collection].aggregate(
                pipeline, maxTimeMS=int(self.timeout_seconds * 1000)
            )
            docs = await cursor.to_list(length=limit)
        except Exception as exc:
            name = type(exc).__name__
            if "Timeout" in name or "ExecutionTimeout" in name:
                raise AdapterTimeout(f"query exceeded {self.timeout_seconds}s") from exc
            raise AdapterError(f"database error: {name}") from exc
        return [normalise_row(entity, ent, d, None) for d in docs]

    # --- orders ----------------------------------------------------------------------
    async def get_order(self, order_id: str, owner: str | None) -> Row | None:
        matches = [self._match_stage("order", "id", _id_variants(str(order_id)))]
        if owner is not None:
            matches.append(self._match_stage("order", "customer_id", _id_variants(str(owner))))
        rows = await self._run("order", self._pipeline("order", matches, limit=1))
        return rows[0] if rows else None

    async def get_order_items(self, order_id: str) -> list[Row]:
        matches = [self._match_stage("order_item", "order_id", _id_variants(str(order_id)))]
        return await self._run("order_item", self._pipeline("order_item", matches))

    async def list_orders(self, owner: str | None, status: str | None, limit: int) -> list[Row]:
        matches: list[dict[str, Any]] = []
        if owner is not None:
            matches.append(self._match_stage("order", "customer_id", _id_variants(str(owner))))
        if status:
            raw = db_statuses_for(self.mapping.entity("order"), status)
            matches.append(self._match_stage("order", "status", raw))
        pipeline = self._pipeline(
            "order", matches, sort=("created_at", -1), limit=min(limit, self.max_rows)
        )
        return await self._run("order", pipeline)

    async def get_shipment(self, order_id: str) -> Row | None:
        matches = [self._match_stage("shipment", "order_id", _id_variants(str(order_id)))]
        sort = ("updated_at", -1) if "updated_at" in self._parsed["shipment"] else None
        rows = await self._run("shipment", self._pipeline("shipment", matches, sort=sort, limit=1))
        return rows[0] if rows else None

    async def get_return_requests(self, order_id: str) -> list[Row]:
        if not self.mapping.has("return_request"):
            return []
        matches = [self._match_stage("return_request", "order_id", _id_variants(str(order_id)))]
        return await self._run("return_request", self._pipeline("return_request", matches))

    # --- catalogue -------------------------------------------------------------------
    async def get_products(self, skus: list[str]) -> dict[str, Row]:
        if not skus:
            return {}
        variants = [v for s in skus for v in _id_variants(str(s))]
        matches = [self._match_stage("product", "sku", variants)]
        rows = await self._run("product", self._pipeline("product", matches))
        return {r["sku"]: r for r in rows}

    async def search_products(
        self,
        *,
        category: str | None,
        min_price: float | None,
        max_price: float | None,
        candidate_cap: int,
    ) -> list[Row]:
        # Filters run in Python so semantics match the SQL adapter exactly.
        rows = await self._run(
            "product", self._pipeline("product", [], limit=min(candidate_cap, 1000))
        )
        out = []
        for r in rows:
            price = r.get("price")
            if category and str(r.get("category", "")).lower() != category.lower():
                continue
            if isinstance(price, int | float):
                if min_price is not None and price < min_price:
                    continue
                if max_price is not None and price > max_price:
                    continue
            out.append(r)
        return out

    async def get_inventory(self, skus: list[str]) -> dict[str, int]:
        if not skus:
            return {}
        variants = [v for s in skus for v in _id_variants(str(s))]
        matches = [self._match_stage("inventory", "sku", variants)]
        out: dict[str, int] = {}
        for r in await self._run("inventory", self._pipeline("inventory", matches)):
            qty = r.get("quantity")
            out[r["sku"]] = out.get(r["sku"], 0) + (int(qty) if isinstance(qty, int | float) else 0)
        return out

    # --- validation ------------------------------------------------------------------
    async def validate(self) -> list[Issue]:
        issues: list[Issue] = []
        try:
            existing = set(await self.db.list_collection_names())
            for name, ent in self.mapping.entities.items():
                if ent.collection not in existing:
                    issues.append(
                        Issue("error", name, f"collection {ent.collection!r} does not exist")
                    )
                    continue
                coll = self.db[ent.collection]
                for field, expr in self._parsed[name].items():
                    for path in columns_in(expr):
                        if await coll.find_one({path: {"$exists": True}}, {"_id": 1}) is None:
                            issues.append(
                                Issue("error", name, f"{field}: no document has the path {path!r}")
                            )
                if "status" in self._parsed[name]:
                    issues += await self._validate_statuses(name)
        except Exception as exc:
            issues.append(Issue("error", "-", f"cannot inspect database: {type(exc).__name__}"))
        return issues

    async def _validate_statuses(self, entity: str) -> list[Issue]:
        ent = self.mapping.entity(entity)
        expr = self._parsed[entity]["status"]
        if not isinstance(expr, Col):
            return []
        values = await self.db[ent.collection].distinct(expr.path)
        known = set(ent.status_map) if ent.status_map else CANONICAL_ORDER_STATUSES
        unmapped = sorted({str(v) for v in values if v is not None} - known)
        if not unmapped:
            return []
        level: Literal["error", "warning"] = "error" if entity == "order" else "warning"
        msg = (
            f"status values not covered by status_map: {unmapped}"
            if level == "error"
            else f"unmapped status values (kept as-is): {unmapped}"
        )
        return [Issue(level, entity, msg)]

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
