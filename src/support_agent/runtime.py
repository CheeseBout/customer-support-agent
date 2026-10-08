"""Assemble the Phase-A pipeline from settings."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING

from support_agent.core.settings import Settings
from support_agent.llm.factory import get_chat_model, get_embeddings
from support_agent.rag.index import VectorStore, make_client
from support_agent.rag.pipeline import SupportRAG
from support_agent.rag.retriever import Reranker, Retriever
from support_agent.rag.sparse import SparseEncoder
from support_agent.tools.client import DomainToolClient

if TYPE_CHECKING:
    from support_agent.agent.agent import SupportAgent


def build_retriever(settings: Settings, store: VectorStore) -> Retriever:
    cfg = settings.app.retrieval
    cache = settings.app.embeddings.cache_dir
    sparse = SparseEncoder(cfg.sparse_model, cache_dir=cache) if cfg.hybrid else None
    reranker = Reranker(cache_dir=cache) if cfg.rerank else None
    return Retriever(store, get_embeddings(settings), cfg, sparse, reranker)


@asynccontextmanager
async def open_store(settings: Settings) -> AsyncIterator[VectorStore]:
    store = VectorStore(make_client(settings), settings.app.retrieval.collection)
    try:
        yield store
    finally:
        store.close()


@asynccontextmanager
async def open_pipeline(
    settings: Settings, *, with_llm: bool = True, with_tools: bool = True
) -> AsyncIterator[SupportRAG]:
    """Qdrant store + (optional) chat model + (optional) MCP tool client, closed on exit."""
    async with open_store(settings) as store:
        retriever = build_retriever(settings, store)
        model = get_chat_model(streaming=False, settings=settings) if with_llm else None
        if with_tools:
            async with DomainToolClient.for_stdio(settings) as tools:
                yield SupportRAG(model=model, retriever=retriever, tools=tools, config=settings.app)
        else:
            yield SupportRAG(model=model, retriever=retriever, tools=None, config=settings.app)


@asynccontextmanager
async def open_agent(
    settings: Settings,
    *,
    with_tools: bool = True,
    checkpoint_url: str | None = None,
    drafts_url: str | None = None,
) -> AsyncIterator[SupportAgent]:
    """The Phase-4 agent: Qdrant + chat model + MCP tools + durable conversation state."""
    from support_agent.agent.agent import SupportAgent
    from support_agent.agent.checkpoint import open_checkpointer
    from support_agent.agent.drafting import DraftProposer
    from support_agent.agent.graph import AgentDeps, build_graph
    from support_agent.drafts.repository import create_repository
    from support_agent.drafts.service import DraftService
    from support_agent.memory.workspace import Workspace

    workspace = Workspace(settings.workspace_dir)

    async with (
        AsyncExitStack() as stack,
        open_store(settings) as store,
        open_checkpointer(checkpoint_url or settings.checkpoint_url) as checkpointer,
    ):
        retriever = build_retriever(settings, store)
        model = get_chat_model(streaming=True, settings=settings)

        async def build(client: DomainToolClient | None) -> SupportAgent:
            drafts = proposer = None
            target = drafts_url or settings.drafts_db_url
            if client is not None and target:
                repo = create_repository(target)
                if drafts_url:  # an explicit (throwaway) store: we make its table too
                    await repo.create_schema()
                drafts = DraftService(repo, settings.app.business_rules)
                proposer = DraftProposer(client, drafts, settings.app.business_rules)
                stack.push_async_callback(repo.close)
            deps = AgentDeps(
                model=model,
                retriever=retriever,
                client=client,
                config=settings.app,
                drafts=drafts,
                proposer=proposer,
                workspace=workspace,
            )
            return SupportAgent(
                graph=build_graph(deps, checkpointer),
                retriever=retriever,
                config=settings.app,
                drafts=drafts,
                workspace=workspace,
            )

        if with_tools:
            async with DomainToolClient.for_stdio(settings) as client:
                yield await build(client)
        else:
            yield await build(None)
