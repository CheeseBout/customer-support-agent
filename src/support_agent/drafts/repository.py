"""Storage for drafts: SQL (PostgreSQL, MySQL, SQLite) and MongoDB.

Both talk to a table/collection the agent owns, through an account that may write ONLY there
(SPEC NFR-002); the read-only account used for the shop's data is a different one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from support_agent.drafts.models import Draft

Order = Literal["newest", "oldest"]


class DuplicateKey(Exception):
    """A draft with this idempotency key (or id) already exists."""


class DraftRepository(Protocol):
    async def create_schema(self) -> None: ...
    async def insert(self, draft: Draft) -> None: ...
    async def get(self, draft_id: str) -> Draft | None: ...
    async def get_by_key(self, key: str) -> Draft | None: ...
    async def list(
        self,
        *,
        customer_id: str | None = None,
        statuses: Sequence[str] | None = None,
        draft_type: str | None = None,
        order_id: str | None = None,
        order: Order = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]: ...
    async def transition(
        self,
        draft_id: str,
        *,
        expected: str,
        to: str,
        now: datetime,
        reviewed_by: str | None = None,
        note: str | None = None,
    ) -> Draft | None:
        """Atomically move `expected` -> `to`. None if the draft was not in `expected`."""
        ...

    async def close(self) -> None: ...


# --- SQL ----------------------------------------------------------------------------

metadata = sa.MetaData()
support_drafts = sa.Table(
    "support_drafts",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("type", sa.String(16), nullable=False),
    sa.Column("customer_id", sa.String(64), nullable=False),
    sa.Column("session_id", sa.String(64), nullable=False),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("payload", sa.JSON, nullable=False),
    sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
    sa.Column("order_id", sa.String(64)),  # denormalised from the payload for lookups
    sa.Column("priority_review", sa.Boolean, nullable=False, default=False),
    sa.Column("reviewed_by", sa.String(64)),
    sa.Column("reviewed_at", sa.DateTime),
    sa.Column("review_note", sa.Text),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("updated_at", sa.DateTime, nullable=False),
    sa.Index("ix_support_drafts_customer_status", "customer_id", "status"),
    sa.Index("ix_support_drafts_status_created", "status", "created_at"),
)


def _naive_utc(moment: datetime | None) -> datetime | None:
    """Columns hold naive UTC: portable across databases, unambiguous by convention."""
    if moment is None:
        return None
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


def _aware(moment: datetime | None) -> datetime | None:
    return None if moment is None else (moment if moment.tzinfo else moment.replace(tzinfo=UTC))


def _row_to_draft(row: Any) -> Draft:
    data = dict(row)
    data.pop("order_id", None)
    for key in ("reviewed_at", "created_at", "updated_at"):
        data[key] = _aware(data[key])
    return Draft.model_validate(data)


class SqlDraftRepository:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    @classmethod
    def from_url(cls, url: str) -> SqlDraftRepository:
        return cls(create_async_engine(url, pool_pre_ping=not url.startswith("sqlite")))

    async def create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def insert(self, draft: Draft) -> None:
        values = draft.model_dump(mode="json")
        values["order_id"] = draft.order_id
        for key in ("reviewed_at", "created_at", "updated_at"):
            values[key] = _naive_utc(getattr(draft, key))
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.insert(support_drafts).values(**values))
        except IntegrityError as exc:
            raise DuplicateKey(draft.idempotency_key) from exc

    async def _one(self, where: sa.ColumnElement[bool]) -> Draft | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(sa.select(support_drafts).where(where))).mappings().first()
        return _row_to_draft(row) if row else None

    async def get(self, draft_id: str) -> Draft | None:
        return await self._one(support_drafts.c.id == draft_id)

    async def get_by_key(self, key: str) -> Draft | None:
        return await self._one(support_drafts.c.idempotency_key == key)

    async def list(
        self,
        *,
        customer_id: str | None = None,
        statuses: Sequence[str] | None = None,
        draft_type: str | None = None,
        order_id: str | None = None,
        order: Order = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]:
        stmt = sa.select(support_drafts)
        if customer_id is not None:
            stmt = stmt.where(support_drafts.c.customer_id == customer_id)
        if statuses:
            stmt = stmt.where(support_drafts.c.status.in_(list(statuses)))
        if draft_type:
            stmt = stmt.where(support_drafts.c.type == draft_type)
        if order_id is not None:
            stmt = stmt.where(support_drafts.c.order_id == order_id)
        created = support_drafts.c.created_at
        stmt = stmt.order_by(created.desc() if order == "newest" else created.asc())
        async with self.engine.connect() as conn:
            rows = (await conn.execute(stmt.limit(limit).offset(offset))).mappings().all()
        return [_row_to_draft(r) for r in rows]

    async def transition(
        self,
        draft_id: str,
        *,
        expected: str,
        to: str,
        now: datetime,
        reviewed_by: str | None = None,
        note: str | None = None,
    ) -> Draft | None:
        stamp = _naive_utc(now)
        values: dict[str, Any] = {"status": to, "updated_at": stamp}
        if to in ("approved", "rejected"):
            values |= {"reviewed_by": reviewed_by, "reviewed_at": stamp, "review_note": note}
        elif note:
            values["review_note"] = note
        async with self.engine.begin() as conn:
            result = await conn.execute(
                sa.update(support_drafts)
                .where(support_drafts.c.id == draft_id, support_drafts.c.status == expected)
                .values(**values)
            )
            if result.rowcount == 0:
                return None
        return await self.get(draft_id)

    async def close(self) -> None:
        await self.engine.dispose()


# --- MongoDB ------------------------------------------------------------------------


class MongoDraftRepository:
    def __init__(self, db: Any, client: Any = None) -> None:
        self.collection = db["support_drafts"]
        self._client = client

    @classmethod
    def from_url(cls, url: str) -> MongoDraftRepository:
        from motor.motor_asyncio import AsyncIOMotorClient

        client: Any = AsyncIOMotorClient(url, tz_aware=True)
        return cls(client.get_default_database(), client)

    async def create_schema(self) -> None:
        await self.collection.create_index("id", unique=True)
        await self.collection.create_index("idempotency_key", unique=True)
        await self.collection.create_index([("customer_id", 1), ("status", 1)])
        await self.collection.create_index([("status", 1), ("created_at", 1)])

    @staticmethod
    def _to_draft(doc: dict[str, Any] | None) -> Draft | None:
        if doc is None:
            return None
        doc = {k: v for k, v in doc.items() if k not in ("_id", "order_id")}
        for key in ("reviewed_at", "created_at", "updated_at"):
            doc[key] = _aware(doc.get(key))
        return Draft.model_validate(doc)

    async def insert(self, draft: Draft) -> None:
        from pymongo.errors import DuplicateKeyError

        doc = draft.model_dump(mode="python")
        doc["order_id"] = draft.order_id
        try:
            await self.collection.insert_one(doc)
        except DuplicateKeyError as exc:
            raise DuplicateKey(draft.idempotency_key) from exc

    async def get(self, draft_id: str) -> Draft | None:
        return self._to_draft(await self.collection.find_one({"id": draft_id}))

    async def get_by_key(self, key: str) -> Draft | None:
        return self._to_draft(await self.collection.find_one({"idempotency_key": key}))

    async def list(
        self,
        *,
        customer_id: str | None = None,
        statuses: Sequence[str] | None = None,
        draft_type: str | None = None,
        order_id: str | None = None,
        order: Order = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]:
        query: dict[str, Any] = {}
        if customer_id is not None:
            query["customer_id"] = customer_id
        if statuses:
            query["status"] = {"$in": list(statuses)}
        if draft_type:
            query["type"] = draft_type
        if order_id is not None:
            query["order_id"] = order_id
        cursor = self.collection.find(query).sort("created_at", -1 if order == "newest" else 1)
        docs = await cursor.skip(offset).limit(limit).to_list(length=limit)
        return [d for d in (self._to_draft(doc) for doc in docs) if d is not None]

    async def transition(
        self,
        draft_id: str,
        *,
        expected: str,
        to: str,
        now: datetime,
        reviewed_by: str | None = None,
        note: str | None = None,
    ) -> Draft | None:
        from pymongo import ReturnDocument

        values: dict[str, Any] = {"status": to, "updated_at": now}
        if to in ("approved", "rejected"):
            values |= {"reviewed_by": reviewed_by, "reviewed_at": now, "review_note": note}
        elif note:
            values["review_note"] = note
        doc = await self.collection.find_one_and_update(
            {"id": draft_id, "status": expected},
            {"$set": values},
            return_document=ReturnDocument.AFTER,
        )
        return self._to_draft(doc)

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()


def create_repository(url: str) -> DraftRepository:
    """Pick the implementation from the URL scheme; adds the async driver for SQL URLs."""
    from support_agent.mcp_db.factory import normalise_db_url

    scheme = url.split(":", 1)[0].lower()
    if scheme.startswith("mongodb"):
        return MongoDraftRepository.from_url(url)
    if scheme.startswith("postgres"):
        return SqlDraftRepository.from_url(normalise_db_url("postgres", url))
    if scheme.startswith("mysql"):
        return SqlDraftRepository.from_url(normalise_db_url("mysql", url))
    if scheme.startswith("sqlite"):
        return SqlDraftRepository.from_url(normalise_db_url("sqlite", url))
    raise ValueError(f"Unsupported DRAFTS_DB_URL scheme: {scheme!r}")
