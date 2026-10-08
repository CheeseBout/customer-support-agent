"""Durable conversation state (SPEC FR-205): SQLite by default, PostgreSQL optionally."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT_URL = "sqlite:///./data/checkpoints.db"


def _on_proactor_loop() -> bool:
    proactor = getattr(asyncio, "ProactorEventLoop", None)  # only exists on Windows
    return proactor is not None and isinstance(asyncio.get_running_loop(), proactor)


def sqlite_path(url: str) -> str:
    """`sqlite:///./x.db`, `sqlite+aiosqlite:///C:/x.db` and bare paths all mean a file."""
    for prefix in ("sqlite+aiosqlite:///", "sqlite:///"):
        if url.startswith(prefix):
            return url[len(prefix) :]
    return url


@asynccontextmanager
async def open_checkpointer(url: str | None = None) -> AsyncIterator[Any]:
    """Yield a LangGraph checkpointer for `url`. `memory` keeps nothing across restarts."""
    target = (url or DEFAULT_CHECKPOINT_URL).strip()

    if target == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
    elif target.startswith("postgres"):
        if _on_proactor_loop():
            raise RuntimeError(
                "PostgreSQL checkpoints do not work on Windows here: psycopg cannot use the "
                "ProactorEventLoop, and the MCP stdio subprocess needs it. Use the default "
                "SQLite checkpoints on Windows, or run on Linux (Docker)."
            )
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "PostgreSQL checkpoints need: pip install 'support-agent[postgres]'"
            ) from exc
        # psycopg wants a plain DSN, not the SQLAlchemy "+asyncpg" form used elsewhere.
        dsn = target.replace("+asyncpg", "").replace("+psycopg", "")
        async with AsyncPostgresSaver.from_conn_string(dsn) as saver:
            await saver.setup()
            yield saver
    else:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        path = Path(sqlite_path(target))
        path.parent.mkdir(parents=True, exist_ok=True)
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            yield saver
