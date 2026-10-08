"""Build the right DataAdapter from settings + mapping."""

from __future__ import annotations

from support_agent.core.settings import Settings
from support_agent.mcp_db.adapters.base import DataAdapter
from support_agent.mcp_db.mapping import SchemaMapping


class ConfigError(RuntimeError):
    pass


def normalise_db_url(db_type: str, url: str) -> str:
    """Accept plain `postgresql://` / `mysql://` / `sqlite:///` and add the async driver."""
    if db_type == "postgres":
        for prefix in ("postgresql://", "postgres://"):
            if url.startswith(prefix):
                return "postgresql+asyncpg://" + url[len(prefix) :]
    elif db_type == "mysql" and url.startswith("mysql://"):
        return "mysql+aiomysql://" + url[len("mysql://") :]
    elif db_type == "sqlite" and url.startswith("sqlite:///") and "+aiosqlite" not in url:
        return "sqlite+aiosqlite:///" + url[len("sqlite:///") :]
    return url


def create_adapter(settings: Settings, mapping: SchemaMapping) -> DataAdapter:
    if not settings.business_db_url:
        raise ConfigError("BUSINESS_DB_URL is not set")
    if mapping.dialect != settings.business_db_type:
        raise ConfigError(
            f"schema mapping dialect is {mapping.dialect!r} but BUSINESS_DB_TYPE is "
            f"{settings.business_db_type!r}"
        )
    kwargs = {
        "max_rows": settings.app.db.max_rows,
        "timeout_seconds": settings.app.db.query_timeout_seconds,
        "timezone": settings.app.business_rules.timezone,
    }
    url = normalise_db_url(settings.business_db_type, settings.business_db_url)
    if settings.business_db_type == "mongodb":
        from support_agent.mcp_db.adapters.mongo import MongoAdapter

        return MongoAdapter.from_url(url, mapping, **kwargs)
    from support_agent.mcp_db.adapters.sql import SqlAdapter

    return SqlAdapter.from_url(url, mapping, **kwargs)
