from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig, Settings, load_app_config
from support_agent.mcp_db.adapters.sql import SqlAdapter
from support_agent.mcp_db.mapping import SchemaMapping, load_mapping
from support_agent.mcp_db.server import build_server
from support_agent.mcp_db.service import BusinessService
from support_agent.seed.demo import seed_sql
from support_agent.tools.client import DomainToolClient

ROOT = Path(__file__).resolve().parent.parent
SECRET = b"test-secret"
TZ = ZoneInfo("Asia/Ho_Chi_Minh")

ALICE = Principal(user_id="u_100")  # owns 1234, 1235, ...
BOB = Principal(user_id="u_101")  # owns 2001, 2002
STAFF = Principal(user_id="s_1", role="staff")


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_app_config(ROOT / "config" / "app.yaml")


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, app_config_path=ROOT / "config" / "app.yaml")


@pytest.fixture(scope="session")
def sqlite_mapping() -> SchemaMapping:
    return load_mapping(ROOT / "config" / "examples" / "schema_mapping.sqlite.yaml")


@pytest_asyncio.fixture
async def sqlite_url(tmp_path: Path) -> str:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'shop.db').as_posix()}"
    await seed_sql(url, now=datetime.now(TZ))
    return url


@pytest_asyncio.fixture
async def adapter(sqlite_url: str, sqlite_mapping: SchemaMapping) -> AsyncIterator[SqlAdapter]:
    a = SqlAdapter.from_url(sqlite_url, sqlite_mapping, timezone="Asia/Ho_Chi_Minh")
    yield a
    await a.close()


@pytest.fixture
def service(adapter: SqlAdapter, app_config: AppConfig) -> BusinessService:
    return BusinessService(adapter, app_config.business_rules)


@pytest_asyncio.fixture
async def tool_client(service: BusinessService) -> AsyncIterator[DomainToolClient]:
    """An in-process MCP client.

    anyio cancel scopes must be exited by the task that entered them, but pytest-asyncio runs
    fixture setup and teardown in different tasks. So one dedicated task owns the client.
    """
    ready, stop = asyncio.Event(), asyncio.Event()
    box: dict[str, DomainToolClient] = {}

    async def owner() -> None:
        async with DomainToolClient(build_server(service, SECRET), SECRET) as client:
            box["client"] = client
            ready.set()
            await stop.wait()

    task = asyncio.create_task(owner())
    await asyncio.wait_for(ready.wait(), timeout=10)
    yield box["client"]
    stop.set()
    await task
