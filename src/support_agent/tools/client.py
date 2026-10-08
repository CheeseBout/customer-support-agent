"""MCP client for the domain tools. Attaches a signed principal to every call."""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.server.mcpserver import MCPServer

from support_agent.core.principal import Principal, sign_principal
from support_agent.core.results import ToolResult
from support_agent.core.settings import Settings
from support_agent.mcp_db.server import PRINCIPAL_META_KEY

log = logging.getLogger(__name__)


class DomainToolClient:
    """Async context manager. `target` is a stdio launch spec, a URL, or (tests) an MCPServer."""

    def __init__(
        self,
        target: StdioServerParameters | MCPServer | str,
        secret: bytes,
        *,
        call_timeout_seconds: float = 20,
    ) -> None:
        self._target = target
        self._secret = secret
        self._timeout = call_timeout_seconds
        self._stack = AsyncExitStack()
        self._client: Client | None = None

    @classmethod
    def for_stdio(cls, settings: Settings) -> DomainToolClient:
        """Spawn `python -m support_agent.mcp_db.server` as a child process."""
        configured = settings.mcp_principal_secret
        secret_text = configured.get_secret_value() if configured else secrets.token_hex(32)
        env = {**os.environ, "MCP_PRINCIPAL_SECRET": secret_text}
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "support_agent.mcp_db.server"], env=env
        )
        return cls(
            params,
            secret_text.encode(),
            call_timeout_seconds=settings.app.agent.run_timeout_seconds,
        )

    async def __aenter__(self) -> DomainToolClient:
        self._client = await self._stack.enter_async_context(Client(self._target))
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()
        self._client = None

    async def call(self, tool: str, args: dict[str, Any], principal: Principal) -> ToolResult:
        if self._client is None:
            raise RuntimeError("DomainToolClient is not connected; use `async with`")
        meta: Any = {PRINCIPAL_META_KEY: sign_principal(principal, self._secret)}
        try:
            result = await self._client.call_tool(
                tool, args, meta=meta, read_timeout_seconds=self._timeout
            )
        except TimeoutError:
            return ToolResult.failure("TIMEOUT", "The tool call timed out.")
        except Exception as exc:
            log.warning("MCP call %s failed: %s", tool, type(exc).__name__)
            return ToolResult.failure("UPSTREAM_ERROR", "The tool service is unavailable.")
        return self._parse(tool, result)

    @staticmethod
    def _parse(tool: str, result: Any) -> ToolResult:
        text = "".join(getattr(b, "text", "") for b in result.content)
        if result.is_error:
            # MCP-level argument validation failure (wrong type, missing field).
            return ToolResult.failure("INVALID_ARGUMENT", f"Invalid arguments for {tool}.")
        try:
            return ToolResult.model_validate(json.loads(text))
        except (ValueError, TypeError):
            log.warning("tool %s returned an unparseable payload", tool)
            return ToolResult.failure("UPSTREAM_ERROR", "The tool returned an invalid response.")
