"""Everything the HTTP layer needs, built once at startup and shared by all requests."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

from support_agent.agent.agent import SupportAgent
from support_agent.api.auth import JwtVerifier
from support_agent.core.principal import Principal
from support_agent.core.settings import Settings
from support_agent.drafts.events import build_dispatcher
from support_agent.drafts.service import DraftService
from support_agent.memory.store import MemoryStore
from support_agent.memory.workspace import Workspace
from support_agent.observability.langfuse import Tracing, create_tracing
from support_agent.rag.ingest import IngestReport
from support_agent.security.guardrails import RateLimiter

log = logging.getLogger(__name__)

READY_CACHE_SECONDS = 30.0
PURGE_INTERVAL_SECONDS = 6 * 3600.0


@dataclass
class Readiness:
    """What `/ready` reports. `llm` is set once at startup: a model call is slow and costs money."""

    checks: dict[str, Callable[[], Awaitable[str | None]]] = field(default_factory=dict)
    llm: str = "pending"
    _cache: tuple[float, dict[str, str]] | None = None

    async def run(self) -> dict[str, str]:
        now = time.monotonic()
        if self._cache and now - self._cache[0] < READY_CACHE_SECONDS:
            return {**self._cache[1], "llm": self.llm}
        results: dict[str, str] = {}
        for name, check in self.checks.items():
            try:
                problem = await check()
            except Exception as exc:
                problem = type(exc).__name__
            results[name] = "ok" if problem is None else f"fail: {problem}"
        self._cache = (now, results)
        return {**results, "llm": self.llm}

    @staticmethod
    def ready(report: dict[str, str]) -> bool:
        return all(v == "ok" or v.startswith(("ok", "degraded")) for v in report.values())


@dataclass
class Services:
    settings: Settings
    agent: SupportAgent
    store: MemoryStore
    workspace: Workspace
    verifier: JwtVerifier
    limiter: RateLimiter
    readiness: Readiness = field(default_factory=Readiness)
    tracing: Tracing = field(default_factory=Tracing)
    ingest: Callable[[bool], Awaitable[IngestReport]] | None = None

    @property
    def drafts(self) -> DraftService | None:
        return self.agent.drafts


async def purge_expired(services: Services) -> int:
    """Drop sessions idle for longer than `memory.session_retention_days` (SPEC 12.3, NFR-011)."""
    days = services.settings.app.memory.session_retention_days
    expired = await services.store.expired_sessions(days)
    for user_id, session_id in expired:
        await services.agent.delete_session(Principal(user_id=user_id), session_id)
        services.workspace.delete_session(user_id, session_id)
        await services.store.delete_session(user_id, session_id)
    services.workspace.purge_older_than(days)
    await services.store.purge_expired_facts()
    if expired:
        log.info("purged %d expired session(s)", len(expired))
    return len(expired)


async def _purge_loop(services: Services) -> None:
    while True:
        try:
            await purge_expired(services)
        except Exception:
            log.exception("session purge failed")
        await asyncio.sleep(PURGE_INTERVAL_SECONDS)


@asynccontextmanager
async def open_services(settings: Settings) -> AsyncIterator[Services]:
    """Build the real services: agent (Qdrant, MCP, model), stores, JWT verifier."""
    from support_agent.llm.capabilities import check_capabilities
    from support_agent.llm.factory import get_chat_model
    from support_agent.rag.ingest import ingest as run_ingest
    from support_agent.runtime import open_agent

    verifier = JwtVerifier(settings)  # fails fast when the JWT settings are unusable
    async with AsyncExitStack() as stack:
        store = await stack.enter_async_context(MemoryStore.open(settings.sessions_db_url))
        agent = await stack.enter_async_context(open_agent(settings))
        retriever = agent.retriever
        ingest_lock = asyncio.Lock()

        async def do_ingest(full: bool) -> IngestReport:
            async with ingest_lock:
                return await asyncio.to_thread(
                    run_ingest,
                    settings.app.knowledge.dir,
                    store=retriever.store,
                    embeddings=retriever.embeddings,
                    sparse=retriever.sparse,
                    embedding_model=settings.embedding_model_name,
                    cfg=settings.app.retrieval,
                    full=full,
                )

        async def qdrant_check() -> str | None:
            count = await asyncio.to_thread(retriever.store.count)
            return None if count else "no documents indexed: run `support-agent ingest`"

        readiness = Readiness(checks={"qdrant": qdrant_check})
        services = Services(
            settings=settings,
            agent=agent,
            store=store,
            workspace=agent.workspace or Workspace(settings.workspace_dir),
            verifier=verifier,
            limiter=RateLimiter(settings.app.guardrails.rate_limit_per_minute),
            readiness=readiness,
            tracing=create_tracing(settings),
            ingest=do_ingest,
        )

        async def probe_llm() -> None:
            try:
                report = await check_capabilities(
                    get_chat_model(streaming=False, settings=settings)
                )
            except Exception as exc:
                readiness.llm = f"fail: {type(exc).__name__}: {exc}"[:200]
                if settings.strict_capability_check:
                    raise
                return
            readiness.llm = report.status
            if report.warnings:
                log.warning("model capability warnings: %s", "; ".join(report.warnings))
            if report.status == "failed" and settings.strict_capability_check:
                raise RuntimeError("the model lacks tool calling or structured output")

        if settings.strict_capability_check:
            await probe_llm()  # refuse to start with a model that cannot do the job
        else:
            task = asyncio.create_task(probe_llm())
            stack.callback(task.cancel)

        purge = asyncio.create_task(_purge_loop(services))
        stack.callback(purge.cancel)
        if services.drafts is not None:
            dispatcher = build_dispatcher(settings, services.drafts.repo)
            if dispatcher is not None:  # tell the shop system about decisions, with retries
                stack.push_async_callback(dispatcher.close)
                delivery = asyncio.create_task(dispatcher.run_forever())
                stack.callback(delivery.cancel)
        yield services
