"""Conversation state in PostgreSQL (PLAN Phase 4: SQLite or Postgres checkpointer)."""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Iterator
from typing import Any

import pytest

from support_agent.agent.agent import SupportAgent
from support_agent.agent.checkpoint import open_checkpointer
from support_agent.agent.graph import AgentDeps, build_graph
from support_agent.core.settings import AppConfig
from tests.conftest import ALICE, BOB
from tests.fakes import AgentFakeLLM

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def postgres_dsn() -> Iterator[str]:
    from testcontainers.community.postgres import PostgresContainer

    try:
        container = PostgresContainer(
            "postgres:16-alpine", username="cp", password="cp", dbname="cp"
        )
        container.start()
    except Exception as exc:
        pytest.skip(f"cannot start postgres: {type(exc).__name__}")
    try:
        host, port = container.get_container_host_ip(), container.get_exposed_port(5432)
        yield f"postgresql://cp:cp@{host}:{port}/cp"
    finally:
        container.stop()


def agent_on(checkpointer: Any, config: AppConfig) -> SupportAgent:
    llm = AgentFakeLLM([], route="chitchat")  # canned replies: no data tools are needed
    deps = AgentDeps(model=llm.model, retriever=None, client=None, config=config)  # type: ignore[arg-type]
    return SupportAgent(graph=build_graph(deps, checkpointer), retriever=None, config=config)  # type: ignore[arg-type]


def test_conversations_survive_a_restart_in_postgres(postgres_dsn: str, app_config: AppConfig):
    async def body() -> None:
        async with open_checkpointer(postgres_dsn) as cp:
            sid = (await agent_on(cp, app_config).answer("hello", ALICE)).session_id
        assert sid

        async with open_checkpointer(postgres_dsn) as cp:  # a new process would do exactly this
            agent = agent_on(cp, app_config)
            assert [h["role"] for h in await agent.history(ALICE, sid)] == ["user", "assistant"]
            await agent.answer("hello again", ALICE, session_id=sid)
            assert len(await agent.history(ALICE, sid)) == 4
            assert await agent.history(BOB, sid) == []  # still isolated per user

    # psycopg's async mode needs a selector loop; Windows defaults to the proactor loop.
    asyncio.run(body(), loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()))
