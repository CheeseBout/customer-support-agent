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

from support_agent.drafts.events import DraftEvent, EventState
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
        updated_after: datetime | None = None,
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

    # The outbox of events for the shop system (see drafts/events.py).
    async def enqueue_event(self, event: DraftEvent) -> bool:
        """Store the event. False if one with this id exists already."""
        ...

    async def due_events(self, now: datetime, limit: int) -> Sequence[DraftEvent]:
        """Pending events whose time has come, oldest first."""
        ...

    async def update_event(
        self,
        event_id: str,
        *,
        state: EventState,
        attempts: int,
        next_attempt_at: datetime,
        last_error: str | None,
        delivered_at: datetime | None,
    ) -> None: ...
    async def list_events(
        self, *, state: str | None = None, limit: int = 50
    ) -> Sequence[DraftEvent]: ...
    async def retry_failed_events(self, now: datetime) -> int:
        """Make every failed event due again. Returns how many."""
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


support_draft_events = sa.Table(
    "support_draft_events",
    metadata,
    sa.Column("id", sa.String(80), primary_key=True),
    sa.Column("draft_id", sa.String(36), nullable=False, index=True),
    sa.Column("event", sa.String(32), nullable=False),
    sa.Column("body", sa.JSON, nullable=False),
    sa.Column("state", sa.String(12), nullable=False),
    sa.Column("attempts", sa.Integer, nullable=False, default=0),
    sa.Column("next_attempt_at", sa.DateTime, nullable=False),
    sa.Column("last_error", sa.String(200)),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("delivered_at", sa.DateTime),
    sa.Index("ix_support_draft_events_due", "state", "next_attempt_at"),
)


def _naive_utc(moment: datetime | None) -> datetime | None:
    """Columns hold naive UTC: portable across databases, unambiguous by convention."""
    if moment is None:
        return None
    return moment.astimezone(UTC).replace(tzinfo=None) if moment.tzinfo else moment


def _aware(moment: datetime | None) -> datetime | None:
    return None if moment is None else (moment if moment.tzinfo else moment.replace(tzinfo=UTC))


def _event_row(row: Any) -> DraftEvent:
    data = dict(row)
    for key in ("next_attempt_at", "created_at", "delivered_at"):
        data[key] = _aware(data[key])
    return DraftEvent.model_validate(data)


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
        updated_after: datetime | None = None,
        order: Order = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]:
        stmt = sa.select(support_drafts)
        if updated_after is not None:
            stmt = stmt.where(support_drafts.c.updated_at >= _naive_utc(updated_after))
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

    async def enqueue_event(self, event: DraftEvent) -> bool:
        values = event.model_dump(mode="json")
        for key in ("next_attempt_at", "created_at", "delivered_at"):
            values[key] = _naive_utc(getattr(event, key))
        try:
            async with self.engine.begin() as conn:
                await conn.execute(sa.insert(support_draft_events).values(**values))
        except IntegrityError:
            return False
        return True

    async def due_events(self, now: datetime, limit: int) -> Sequence[DraftEvent]:
        stmt = (
            sa.select(support_draft_events)
            .where(
                support_draft_events.c.state == "pending",
                support_draft_events.c.next_attempt_at <= _naive_utc(now),
            )
            .order_by(support_draft_events.c.created_at.asc(), support_draft_events.c.id.asc())
            .limit(limit)
        )
        async with self.engine.connect() as conn:
            return [_event_row(r) for r in (await conn.execute(stmt)).mappings().all()]

    async def update_event(
        self,
        event_id: str,
        *,
        state: EventState,
        attempts: int,
        next_attempt_at: datetime,
        last_error: str | None,
        delivered_at: datetime | None,
    ) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                sa.update(support_draft_events)
                .where(support_draft_events.c.id == event_id)
                .values(
                    state=state,
                    attempts=attempts,
                    next_attempt_at=_naive_utc(next_attempt_at),
                    last_error=last_error,
                    delivered_at=_naive_utc(delivered_at),
                )
            )

    async def list_events(
        self, *, state: str | None = None, limit: int = 50
    ) -> Sequence[DraftEvent]:
        stmt = sa.select(support_draft_events)
        if state:
            stmt = stmt.where(support_draft_events.c.state == state)
        stmt = stmt.order_by(support_draft_events.c.created_at.desc()).limit(limit)
        async with self.engine.connect() as conn:
            return [_event_row(r) for r in (await conn.execute(stmt)).mappings().all()]

    async def retry_failed_events(self, now: datetime) -> int:
        async with self.engine.begin() as conn:
            result = await conn.execute(
                sa.update(support_draft_events)
                .where(support_draft_events.c.state == "failed")
                .values(state="pending", attempts=0, next_attempt_at=_naive_utc(now))
            )
        return int(result.rowcount or 0)

    async def close(self) -> None:
        await self.engine.dispose()


# --- MongoDB ------------------------------------------------------------------------


class MongoDraftRepository:
    def __init__(self, db: Any, client: Any = None) -> None:
        self.collection = db["support_drafts"]
        self.events = db["support_draft_events"]
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
        await self.events.create_index("id", unique=True)
        await self.events.create_index([("state", 1), ("next_attempt_at", 1)])

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
        updated_after: datetime | None = None,
        order: Order = "newest",
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]:
        query: dict[str, Any] = {}
        if updated_after is not None:
            query["updated_at"] = {"$gte": updated_after}
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

    @staticmethod
    def _to_event(doc: dict[str, Any]) -> DraftEvent:
        data = {k: v for k, v in doc.items() if k != "_id"}
        for key in ("next_attempt_at", "created_at", "delivered_at"):
            data[key] = _aware(data.get(key))
        return DraftEvent.model_validate(data)

    async def enqueue_event(self, event: DraftEvent) -> bool:
        from pymongo.errors import DuplicateKeyError

        try:
            await self.events.insert_one(event.model_dump(mode="python"))
        except DuplicateKeyError:
            return False
        return True

    async def due_events(self, now: datetime, limit: int) -> Sequence[DraftEvent]:
        cursor = self.events.find({"state": "pending", "next_attempt_at": {"$lte": now}})
        docs = await cursor.sort([("created_at", 1), ("id", 1)]).limit(limit).to_list(length=limit)
        return [self._to_event(d) for d in docs]

    async def update_event(
        self,
        event_id: str,
        *,
        state: EventState,
        attempts: int,
        next_attempt_at: datetime,
        last_error: str | None,
        delivered_at: datetime | None,
    ) -> None:
        await self.events.update_one(
            {"id": event_id},
            {
                "$set": {
                    "state": state,
                    "attempts": attempts,
                    "next_attempt_at": next_attempt_at,
                    "last_error": last_error,
                    "delivered_at": delivered_at,
                }
            },
        )

    async def list_events(
        self, *, state: str | None = None, limit: int = 50
    ) -> Sequence[DraftEvent]:
        cursor = self.events.find({"state": state} if state else {}).sort("created_at", -1)
        return [self._to_event(d) for d in await cursor.limit(limit).to_list(length=limit)]

    async def retry_failed_events(self, now: datetime) -> int:
        result = await self.events.update_many(
            {"state": "failed"},
            {"$set": {"state": "pending", "attempts": 0, "next_attempt_at": now}},
        )
        return int(result.modified_count)

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
