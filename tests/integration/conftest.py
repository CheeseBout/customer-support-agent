"""Real databases via testcontainers. Run with: pytest -m integration (needs Docker)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.mcp_db.adapters.base import DataAdapter
from support_agent.mcp_db.adapters.mongo import MongoAdapter
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.mapping import SchemaMapping, load_mapping
from support_agent.seed.demo import seed_mongo, seed_sql
from tests.conftest import DEMO, TZ

pytestmark = pytest.mark.integration

MAPPINGS = {
    "postgres": DEMO / "config" / "schema_mapping.postgres.yaml",
    "mysql": DEMO / "config" / "schema_mapping.mysql.yaml",
    "mongodb": DEMO / "config" / "schema_mapping.mongodb.yaml",
}


class Deployment:
    """A started database plus the two URLs the product uses: owner (seeding) and read-only."""

    def __init__(
        self, kind: str, owner_url: str, readonly_url: str, mapping: SchemaMapping
    ) -> None:
        self.kind, self.owner_url, self.readonly_url, self.mapping = (
            kind,
            owner_url,
            readonly_url,
            mapping,
        )


def _run(coro: Any) -> Any:
    return asyncio.new_event_loop().run_until_complete(coro)


def _start_postgres() -> Iterator[tuple[str, str, str]]:
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer(
        "postgres:16-alpine", username="owner", password="owner", dbname="shop"
    ) as pg:
        host, port = pg.get_container_host_ip(), pg.get_exposed_port(5432)
        owner = f"postgresql+asyncpg://owner:owner@{host}:{port}/shop"
        yield owner, f"postgresql+asyncpg://support_ro:ro@{host}:{port}/shop", owner


def _start_mysql() -> Iterator[tuple[str, str, str]]:
    from testcontainers.community.mysql import MySqlContainer

    with MySqlContainer(
        "mysql:8.4", username="owner", password="owner", dbname="shop", root_password="root"
    ) as my:
        host, port = my.get_container_host_ip(), my.get_exposed_port(3306)
        yield (
            f"mysql+aiomysql://owner:owner@{host}:{port}/shop",
            f"mysql+aiomysql://support_ro:ro@{host}:{port}/shop",
            f"mysql+aiomysql://root:root@{host}:{port}/shop",  # only root may CREATE USER
        )


def _start_mongo() -> Iterator[tuple[str, str, str]]:
    from testcontainers.community.mongodb import MongoDbContainer

    with MongoDbContainer("mongo:7", username="owner", password="owner") as mongo:
        host, port = mongo.get_container_host_ip(), mongo.get_exposed_port(27017)
        url = f"mongodb://owner:owner@{host}:{port}/shop?authSource=admin"
        yield url, url, url  # read-only role setup is out of scope for the Mongo container


async def _grant_readonly(kind: str, admin_url: str) -> None:
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            if kind == "postgres":
                for stmt in (
                    "CREATE ROLE support_ro LOGIN PASSWORD 'ro'",
                    "GRANT CONNECT ON DATABASE shop TO support_ro",
                    "GRANT USAGE ON SCHEMA public TO support_ro",
                    "GRANT SELECT ON ALL TABLES IN SCHEMA public TO support_ro",
                ):
                    await conn.execute(sa.text(stmt))
            else:
                await conn.execute(sa.text("CREATE USER 'support_ro'@'%' IDENTIFIED BY 'ro'"))
                await conn.execute(sa.text("GRANT SELECT ON shop.* TO 'support_ro'@'%'"))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session", params=["postgres", "mysql", "mongodb"])
def deployment(request: pytest.FixtureRequest) -> Iterator[Deployment]:
    kind: str = request.param
    starter = {"postgres": _start_postgres, "mysql": _start_mysql, "mongodb": _start_mongo}[kind]
    gen = starter()
    try:
        owner_url, readonly_url, admin_url = next(gen)
    except Exception as exc:  # Docker unavailable or image pull failed
        pytest.skip(f"cannot start {kind}: {type(exc).__name__}: {exc}")
    try:
        if kind == "mongodb":
            _run(seed_mongo(owner_url, now=datetime.now(TZ)))
        else:
            _run(seed_sql(owner_url, now=datetime.now(TZ)))
            _run(_grant_readonly(kind, admin_url))
        yield Deployment(kind, owner_url, readonly_url, load_mapping(MAPPINGS[kind]))
    finally:
        gen.close()


@pytest_asyncio.fixture
async def any_adapter(deployment: Deployment) -> AsyncIterator[DataAdapter]:
    """Overrides the unit-test fixture of the same name so the shared contract tests run here."""
    if deployment.kind == "mongodb":
        adapter: DataAdapter = MongoAdapter.from_url(
            deployment.readonly_url, deployment.mapping, timezone="Asia/Ho_Chi_Minh"
        )
    else:
        adapter = SqlAdapter.from_url(
            deployment.readonly_url, deployment.mapping, timezone="Asia/Ho_Chi_Minh"
        )
    yield adapter
    await adapter.close()


@pytest.fixture
def knowledge_dir() -> Path:
    return DEMO / "knowledge"
