"""Small SQLite store for what the checkpointer does not index: sessions, feedback, facts.

* `sessions`: which conversations a user owns (the checkpointer cannot list by user).
* `feedback`: ratings attached to an answer (`POST /v1/feedback`).
* `facts`: long-term preferences a customer agreed to keep (SPEC 12.2), each with an expiry.

Every query is scoped by `user_id`, so one user can never read or delete another's rows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel

from support_agent.agent.checkpoint import sqlite_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    user_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);
"""

TITLE_CHARS = 80


class SessionInfo(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class Fact(BaseModel):
    key: str
    value: str
    created_at: datetime
    expires_at: datetime


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


class MemoryStore:
    def __init__(
        self, db: aiosqlite.Connection, *, clock: Callable[[], datetime] = _utcnow
    ) -> None:
        self._db = db
        self._clock = clock

    @classmethod
    @asynccontextmanager
    async def open(
        cls, url: str, *, clock: Callable[[], datetime] = _utcnow
    ) -> AsyncIterator[MemoryStore]:
        """`sqlite:///path.db` (a file) or `memory` (gone when the process ends)."""
        target = ":memory:"
        if url.strip() != "memory":
            path = Path(sqlite_path(url.strip()))
            path.parent.mkdir(parents=True, exist_ok=True)
            target = str(path)
        async with aiosqlite.connect(target) as db:
            db.row_factory = aiosqlite.Row
            await db.executescript(_SCHEMA)
            await db.commit()
            yield cls(db, clock=clock)

    # --- sessions --------------------------------------------------------------------
    async def touch_session(self, user_id: str, session_id: str, first_message: str = "") -> None:
        now = _iso(self._clock())
        title = " ".join(first_message.split())[:TITLE_CHARS]
        await self._db.execute(
            "INSERT INTO sessions (user_id, session_id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, session_id) DO UPDATE SET "
            "updated_at = excluded.updated_at, "
            "title = CASE WHEN sessions.title = '' THEN excluded.title ELSE sessions.title END",
            (user_id, session_id, title, now, now),
        )
        await self._db.commit()

    async def get_session(self, user_id: str, session_id: str) -> SessionInfo | None:
        async with self._db.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions "
            "WHERE user_id = ? AND session_id = ?",
            (user_id, session_id),
        ) as cur:
            row = await cur.fetchone()
        return SessionInfo(**dict(row)) if row else None

    async def owns_session(self, user_id: str, session_id: str) -> bool:
        return await self.get_session(user_id, session_id) is not None

    async def list_sessions(
        self, user_id: str, *, limit: int = 20, offset: int = 0
    ) -> list[SessionInfo]:
        async with self._db.execute(
            "SELECT session_id, title, created_at, updated_at FROM sessions WHERE user_id = ? "
            "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (user_id, max(1, min(limit, 100)), max(0, offset)),
        ) as cur:
            return [SessionInfo(**dict(row)) async for row in cur]

    async def delete_session(self, user_id: str, session_id: str) -> bool:
        cur = await self._db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND session_id = ?", (user_id, session_id)
        )
        await self._db.commit()
        return cur.rowcount > 0

    async def expired_sessions(self, retention_days: int) -> list[tuple[str, str]]:
        """(user_id, session_id) pairs idle for longer than the retention period."""
        cutoff = _iso(self._clock() - timedelta(days=retention_days))
        async with self._db.execute(
            "SELECT user_id, session_id FROM sessions WHERE updated_at < ?", (cutoff,)
        ) as cur:
            return [(row["user_id"], row["session_id"]) async for row in cur]

    # --- feedback --------------------------------------------------------------------
    async def add_feedback(
        self, user_id: str, session_id: str, message_id: str, rating: int, comment: str = ""
    ) -> None:
        await self._db.execute(
            "INSERT INTO feedback (user_id, session_id, message_id, rating, comment, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, session_id, message_id, rating, comment, _iso(self._clock())),
        )
        await self._db.commit()

    async def feedback_for(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        async with self._db.execute(
            "SELECT message_id, rating, comment, created_at FROM feedback "
            "WHERE user_id = ? AND session_id = ? ORDER BY id",
            (user_id, session_id),
        ) as cur:
            return [dict(row) async for row in cur]

    # --- long-term facts -------------------------------------------------------------
    async def set_fact(self, user_id: str, key: str, value: str, *, ttl_days: int) -> Fact:
        now = self._clock()
        fact = Fact(key=key, value=value, created_at=now, expires_at=now + timedelta(days=ttl_days))
        await self._db.execute(
            "INSERT INTO facts (user_id, key, value, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (user_id, key) DO UPDATE SET "
            "value = excluded.value, created_at = excluded.created_at, "
            "expires_at = excluded.expires_at",
            (user_id, key, value, _iso(fact.created_at), _iso(fact.expires_at)),
        )
        await self._db.commit()
        return fact

    async def list_facts(self, user_id: str) -> list[Fact]:
        """Facts that have not expired. Expired rows are removed on the way."""
        now = _iso(self._clock())
        await self._db.execute("DELETE FROM facts WHERE expires_at <= ?", (now,))
        await self._db.commit()
        async with self._db.execute(
            "SELECT key, value, created_at, expires_at FROM facts WHERE user_id = ? ORDER BY key",
            (user_id,),
        ) as cur:
            return [Fact(**dict(row)) async for row in cur]

    async def purge_expired_facts(self) -> int:
        cur = await self._db.execute(
            "DELETE FROM facts WHERE expires_at <= ?", (_iso(self._clock()),)
        )
        await self._db.commit()
        return cur.rowcount

    async def delete_facts(self, user_id: str, key: str | None = None) -> int:
        if key is None:
            cur = await self._db.execute("DELETE FROM facts WHERE user_id = ?", (user_id,))
        else:
            cur = await self._db.execute(
                "DELETE FROM facts WHERE user_id = ? AND key = ?", (user_id, key)
            )
        await self._db.commit()
        return cur.rowcount
