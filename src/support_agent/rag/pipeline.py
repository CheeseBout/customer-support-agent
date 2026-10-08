"""End-to-end Phase-A pipeline: route -> (policy RAG | personal tools | both) -> grounded answer."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from support_agent.core.i18n import Lang, resolve_language, t
from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig
from support_agent.llm.usage import Usage, UsageTracker
from support_agent.rag.answer import Citation, answer_grounded
from support_agent.rag.personal import PersonalResult, gather_personal
from support_agent.rag.retriever import RetrievedChunk, Retriever
from support_agent.rag.router import Route, route_query
from support_agent.tools.client import DomainToolClient

log = logging.getLogger(__name__)


Outcome = Literal["answered", "clarify", "no_info", "not_found", "refused", "error", "confirmation"]


class RetrievalTrace(BaseModel):
    doc_id: str
    section: str
    score: float


class RagAnswer(BaseModel):
    answer: str
    language: Lang
    route: Route | None = None
    confidence: float | None = None
    citations: list[Citation] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    order_ids: list[str] = Field(default_factory=list)
    skus: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    retrieved: list[RetrievalTrace] = Field(default_factory=list)
    sufficient: bool = True
    outcome: Outcome = "answered"
    latency_ms: int = 0
    steps: int = 0  # agent loop iterations (0 for the Phase-A pipeline)
    session_id: str | None = None  # set by the agent; the Phase-A pipeline is stateless
    trace_id: str | None = None
    drafts: list[dict[str, Any]] = Field(default_factory=list)  # requests created this turn
    interrupt: dict[str, Any] | None = None  # a confirmation the customer still owes


class SupportRAG:
    """Phase-A question answering. `tools` may be None for policy-only deployments."""

    def __init__(
        self,
        *,
        model: BaseChatModel | None,
        retriever: Retriever,
        tools: DomainToolClient | None,
        config: AppConfig,
    ) -> None:
        self.model = model
        self.retriever = retriever
        self.tools = tools
        self.config = config

    async def answer(
        self,
        question: str,
        principal: Principal,
        *,
        language: str | None = "auto",
        history: list[BaseMessage] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> RagAnswer:
        """Answer one question. `callbacks`/`metadata` let callers attach tracing."""
        started = time.perf_counter()
        question = question.strip()[: self.config.guardrails.max_input_chars]
        lang = resolve_language(language, question)
        tracker = UsageTracker()
        run_config: RunnableConfig = {
            "callbacks": [tracker, *(callbacks or [])],
            "metadata": metadata or {},
            "run_name": "support_agent",
        }

        try:
            result = await asyncio.wait_for(
                self._run(question, principal, lang, history, run_config),
                timeout=self.config.agent.run_timeout_seconds,
            )
        except TimeoutError:
            result = RagAnswer(answer=t("timeout", lang), language=lang, outcome="error")
        except Exception:
            log.exception("pipeline failed")
            result = RagAnswer(answer=t("upstream_error", lang), language=lang, outcome="error")

        result.usage = tracker.usage
        result.trace_id = trace_id
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ------------------------------------------------------------------------------------------
    @staticmethod
    def _reply(base: RagAnswer, text: str, outcome: Outcome, **changes: Any) -> RagAnswer:
        return base.model_copy(update={"answer": text, "outcome": outcome, **changes})

    async def _run(
        self,
        question: str,
        principal: Principal,
        lang: Lang,
        history: list[BaseMessage] | None,
        run_config: RunnableConfig,
    ) -> RagAnswer:
        decision = await route_query(
            self.model, question, self.config.router, history=history, config=run_config
        )
        base = RagAnswer(
            answer="",
            language=lang,
            route=decision.route,
            confidence=decision.confidence,
            order_ids=decision.entities.order_ids,
            skus=decision.entities.skus,
        )

        if decision.confidence < self.config.router.min_confidence:
            return self._reply(base, t("clarify", lang), "clarify")
        if decision.route == "chitchat":
            return self._reply(base, t("chitchat", lang), "refused")
        if decision.route == "out_of_scope":
            return self._reply(base, t("out_of_scope", lang), "refused")

        need_policy = decision.route in ("policy", "combined")
        need_personal = decision.route in ("personal", "combined")
        if need_personal and self.tools is None:
            return self._reply(base, t("upstream_error", lang), "error")

        async def policy() -> list[RetrievedChunk]:
            return await self.retriever.retrieve(question) if need_policy else []

        async def personal() -> PersonalResult:
            if not need_personal or self.tools is None:
                return PersonalResult()
            return await gather_personal(
                self.tools,
                principal,
                self.model,
                question,
                decision.entities,
                history=history,
                max_steps=min(4, self.config.agent.max_steps),
                config=run_config,
            )

        # combined runs both halves concurrently (PLAN Phase 2)
        chunks, data = await asyncio.gather(policy(), personal())
        base = base.model_copy(
            update={
                "tool_calls": data.tool_calls,
                "retrieved": [
                    RetrievalTrace(doc_id=c.doc_id, section=c.section, score=c.score)
                    for c in chunks
                ],
            }
        )

        if data.error:
            return self._reply(base, t("upstream_error", lang), "error")

        # Asked only about orders and none belong to the caller: identical reply whether or
        # not the order exists, and no LLM call (so nothing can leak through wording).
        if decision.route == "personal" and data.orders_not_found and not data.has_data:
            ids = ", ".join(f"#{o}" for o in data.orders_not_found)
            return self._reply(
                base, t("order_not_found", lang, order_id=ids), "not_found", sufficient=False
            )

        if self.model is None:
            return self._reply(base, t("upstream_error", lang), "error")

        grounded = await answer_grounded(
            self.model,
            question,
            lang,
            chunks=chunks,
            facts=data.facts,
            history=history,
            config=run_config,
        )
        return self._reply(
            base,
            grounded.text,
            "answered" if grounded.sufficient else "no_info",
            citations=grounded.citations,
            sufficient=grounded.sufficient,
        )
