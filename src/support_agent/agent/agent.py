"""SupportAgent: runs one conversational turn and streams it (SPEC 11, 14.5)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import weakref
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from support_agent.agent.events import AgentEvent
from support_agent.agent.prompts import PROMPT_VERSION
from support_agent.core.i18n import Lang, resolve_language, t
from support_agent.core.principal import Principal, hash_user_id, session_namespace
from support_agent.core.settings import AppConfig
from support_agent.drafts.service import DraftService
from support_agent.llm.structured import message_text
from support_agent.llm.usage import UsageTracker
from support_agent.memory.workspace import Workspace
from support_agent.rag.answer import Citation
from support_agent.rag.pipeline import RagAnswer, RetrievalTrace
from support_agent.rag.retriever import Retriever
from support_agent.security.guardrails import check_input

log = logging.getLogger(__name__)


class SupportAgent:
    """Same call shape as the Phase-A `SupportRAG`, so evaluation can compare the two."""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        retriever: Retriever,
        config: AppConfig,
        drafts: DraftService | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        self.graph = graph
        self.drafts = drafts
        self.workspace = workspace
        self.retriever = retriever
        self.config = config
        # One turn at a time per conversation: two concurrent turns would interleave messages.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    # --- public API ------------------------------------------------------------------
    async def stream(
        self,
        question: str,
        principal: Principal,
        *,
        session_id: str | None = None,
        language: str | None = "auto",
        history: list[BaseMessage] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        preferences: list[tuple[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Events of one turn. Always ends with exactly one `done` or `error` event."""
        started = time.perf_counter()
        verdict = check_input(question, self.config.guardrails)
        question = verdict.text
        lang: Lang = resolve_language(language, question)
        session_id = session_id or uuid.uuid4().hex
        message_id = uuid.uuid4().hex
        thread_id = f"{session_namespace(principal.user_id)}:{session_id}"
        tracker = UsageTracker()
        run_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            # Each loop step is an agent superstep plus a tools superstep, plus fixed nodes.
            "recursion_limit": self.config.agent.max_steps * 2 + 10,
            "callbacks": [tracker, *(callbacks or [])],
            "metadata": {
                "prompt_version": PROMPT_VERSION,
                "session_id": session_id,
                "user": hash_user_id(principal.user_id),
                **(metadata or {}),
            },
            "run_name": "support_agent",
        }
        payload = {
            "messages": [*(history or []), HumanMessage(content=question)],
            "principal": {"user_id": principal.user_id, "role": principal.role},
            "session_id": session_id,
            "language": lang,
            "preferences": [[k, v] for k, v in (preferences or [])],
        }

        yield AgentEvent(kind="session", data={"session_id": session_id, "message_id": message_id})

        if verdict.blocked:
            # Refuse politely, keep the session usable, and keep the attempt out of the history.
            log.warning("input blocked by guardrails", extra={"reasons": ",".join(verdict.reasons)})
            text = t("guardrail_blocked", lang)
            yield AgentEvent(kind="token", data={"text": text})
            values = {
                "answer": text,
                "outcome": "refused",
                "route": "out_of_scope",
                "language": lang,
                "guard_flags": [f"input:{r}" for r in verdict.reasons],
            }
            yield AgentEvent(
                kind="done",
                data=self._done_data(values, tracker, message_id, session_id, trace_id, started),
            )
            return

        async with self._lock(thread_id):
            # A new message while a confirmation is open means the customer moved on.
            await self._abandon_pending(run_config, lang)
            async for event in self._drive(
                payload, run_config, lang, tracker, message_id, session_id, trace_id, started
            ):
                yield event

    async def resume(
        self,
        principal: Principal,
        session_id: str,
        interrupt_id: str,
        decision: str,
        *,
        edits: dict[str, Any] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Answer the confirmation a turn stopped at, then stream the rest of that turn.

        `decision` is approve, reject or edit. An edit stops the turn again with a new
        confirmation (a new id), so the caller always answers the latest one.
        """
        started = time.perf_counter()
        message_id = uuid.uuid4().hex
        thread_id = f"{session_namespace(principal.user_id)}:{session_id}"
        tracker = UsageTracker()
        run_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.config.agent.max_steps * 2 + 10,
            "callbacks": [tracker, *(callbacks or [])],
            "metadata": {
                "prompt_version": PROMPT_VERSION,
                "session_id": session_id,
                "user": hash_user_id(principal.user_id),
                **(metadata or {}),
            },
            "run_name": "support_agent_resume",
        }
        yield AgentEvent(kind="session", data={"session_id": session_id, "message_id": message_id})
        async with self._lock(thread_id):
            waiting = await self._open_confirmation(run_config)
            values = (await self.graph.aget_state(run_config)).values or {}
            lang: Lang = values.get("language", "en")
            if waiting is None or waiting.get("id") != interrupt_id:
                yield AgentEvent(
                    kind="error",
                    data={
                        "code": "NO_PENDING_CONFIRMATION",
                        "message": t("stale_confirmation", lang),
                    },
                )
                return
            command: Command[Any] = Command(
                resume={"interrupt_id": interrupt_id, "decision": decision, "edits": edits or {}}
            )
            async for event in self._drive(
                command, run_config, lang, tracker, message_id, session_id, trace_id, started
            ):
                yield event

    async def pending_confirmation(
        self, principal: Principal, session_id: str
    ) -> dict[str, Any] | None:
        """The confirmation this conversation is waiting on, if any (for reconnecting clients)."""
        thread_id = f"{session_namespace(principal.user_id)}:{session_id}"
        return await self._open_confirmation({"configurable": {"thread_id": thread_id}})

    async def _drive(
        self,
        graph_input: Any,
        run_config: RunnableConfig,
        lang: Lang,
        tracker: UsageTracker,
        message_id: str,
        session_id: str,
        trace_id: str | None,
        started: float,
    ) -> AsyncIterator[AgentEvent]:
        """Run the graph until it finishes or stops for the customer."""
        failure: AgentEvent | None = None
        events: Any = None
        deadline = time.monotonic() + self.config.agent.run_timeout_seconds
        try:
            events = self.graph.astream(graph_input, run_config, stream_mode="custom").__aiter__()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                try:
                    item = await asyncio.wait_for(events.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                yield AgentEvent(**item)
        except TimeoutError:
            failure = AgentEvent(
                kind="error", data={"code": "TIMEOUT", "message": t("timeout", lang)}
            )
        except GraphRecursionError:
            failure = AgentEvent(
                kind="error", data={"code": "STEP_LIMIT", "message": t("step_limit", lang)}
            )
        except Exception:
            log.exception("agent turn failed")
            failure = AgentEvent(
                kind="error",
                data={"code": "UPSTREAM_ERROR", "message": t("upstream_error", lang)},
            )
        finally:
            if events is not None:
                await _close(events)

        if failure is not None:
            await self._repair(run_config, str(failure.data["message"]))
            yield failure
            return

        waiting = await self._open_confirmation(run_config)
        if waiting is not None:
            values = (await self.graph.aget_state(run_config)).values
            turn = self._done_data(values, tracker, message_id, session_id, trace_id, started)
            yield AgentEvent(
                kind="interrupt",
                # `turn` is what happened up to this point (route, tools, usage): the client
                # may ignore it, evaluation needs it.
                data={**waiting, "session_id": session_id, "message_id": message_id, "turn": turn},
            )
            return

        values = (await self.graph.aget_state(run_config)).values
        yield AgentEvent(
            kind="done",
            data=self._done_data(values, tracker, message_id, session_id, trace_id, started),
        )

    def _lock(self, thread_id: str) -> asyncio.Lock:
        lock = self._locks.get(thread_id)
        if lock is None:
            lock = self._locks[thread_id] = asyncio.Lock()
        return lock

    async def _open_confirmation(self, run_config: RunnableConfig) -> dict[str, Any] | None:
        snapshot = await self.graph.aget_state(run_config)
        for task in snapshot.tasks:
            for stop in task.interrupts:
                if isinstance(stop.value, dict) and stop.value.get("kind") == "confirm_draft":
                    return dict(stop.value)
        return None

    async def _abandon_pending(self, run_config: RunnableConfig, lang: Lang) -> None:
        """Close a confirmation nobody answered. Nothing was created, so nothing to undo."""
        if await self._open_confirmation(run_config) is None:
            return
        await self._repair(run_config, t("confirmation_abandoned", lang), pending=None)

    async def answer(
        self,
        question: str,
        principal: Principal,
        *,
        session_id: str | None = None,
        language: str | None = "auto",
        history: list[BaseMessage] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> RagAnswer:
        """Run a turn to completion. Same result type as the Phase-A pipeline."""
        lang: Lang = resolve_language(language, question)
        async for event in self.stream(
            question,
            principal,
            session_id=session_id,
            language=language,
            history=history,
            callbacks=callbacks,
            metadata=metadata,
            trace_id=trace_id,
        ):
            if event.kind == "done":
                return _to_answer(event.data)
            if event.kind == "interrupt":
                return _waiting_answer(event.data, lang)
            if event.kind == "error":
                return RagAnswer(answer=str(event.data["message"]), language=lang, outcome="error")
        return RagAnswer(answer=t("upstream_error", lang), language=lang, outcome="error")

    async def decide(
        self,
        principal: Principal,
        shown: RagAnswer,
        decision: str,
        *,
        edits: dict[str, Any] | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        trace_id: str | None = None,
    ) -> RagAnswer:
        """Answer the confirmation in `shown` and run the turn to its end (or its next stop).

        Same result type as `answer()`, so a caller that only wants outcomes can drive the whole
        request flow without handling events. The confirmation that was answered stays in
        `interrupt` unless the customer was asked again.
        """
        stop = shown.interrupt or {}
        async for event in self.resume(
            principal,
            str(stop.get("session_id")),
            str(stop.get("id")),
            decision,
            edits=edits,
            callbacks=callbacks,
            trace_id=trace_id,
        ):
            if event.kind == "done":
                final = _to_answer(event.data)
                usage = shown.usage.model_copy()
                usage.add(final.usage)
                return final.model_copy(update={"interrupt": stop, "usage": usage})
            if event.kind == "interrupt":
                again = _waiting_answer(event.data, shown.language)
                usage = shown.usage.model_copy()
                usage.add(again.usage)
                return again.model_copy(update={"usage": usage})
            if event.kind == "error":
                return RagAnswer(
                    answer=str(event.data["message"]), language=shown.language, outcome="error"
                )
        return RagAnswer(
            answer=t("upstream_error", shown.language), language=shown.language, outcome="error"
        )

    async def history(self, principal: Principal, session_id: str) -> list[dict[str, str]]:
        """The visible conversation of a session, for the calling user only."""
        thread_id = f"{session_namespace(principal.user_id)}:{session_id}"
        state = await self.graph.aget_state({"configurable": {"thread_id": thread_id}})
        messages = (state.values or {}).get("messages", [])
        return [
            {
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": message_text(m),
            }
            for m in messages
            if isinstance(m, HumanMessage | AIMessage)
        ]

    async def delete_session(self, principal: Principal, session_id: str) -> None:
        """Forget a conversation (SPEC FR-401, retention). Only the caller's own can be reached."""
        thread_id = f"{session_namespace(principal.user_id)}:{session_id}"
        async with self._lock(thread_id):
            await self.graph.checkpointer.adelete_thread(thread_id)  # type: ignore[union-attr]

    # --- internals -------------------------------------------------------------------
    async def _repair(self, run_config: RunnableConfig, text: str, **extra: Any) -> None:
        """After a failed turn, replace its scratch work with the error reply.

        Otherwise the next turn would find a question that never got an answer, followed by
        half-finished tool calls.
        """
        try:
            values = (await self.graph.aget_state(run_config)).values
            messages: list[BaseMessage] = values.get("messages", [])
            turn_id = values.get("turn_human_id")
            index = next((i for i, m in enumerate(messages) if m.id == turn_id), None)
            if index is None:
                return  # the turn failed before it was recorded: nothing to repair
            scratch = [RemoveMessage(id=m.id) for m in messages[index + 1 :] if m.id]
            await self.graph.aupdate_state(
                run_config,
                {
                    "messages": [*scratch, AIMessage(content=text, id=uuid.uuid4().hex)],
                    "pending": None,
                    **extra,
                },
                as_node="finalize",
            )
        except Exception:  # repairing is best-effort; never mask the original failure
            log.exception("could not repair the conversation after a failed turn")

    def _done_data(
        self,
        values: dict[str, Any],
        tracker: UsageTracker,
        message_id: str,
        session_id: str,
        trace_id: str | None,
        started: float,
    ) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "message_id": message_id,
            "answer": values.get("answer", ""),
            "outcome": values.get("outcome") or "error",
            "route": values.get("route"),
            "confidence": values.get("confidence"),
            "language": values.get("language", "en"),
            "citations": values.get("citations", []),
            "order_ids": values.get("order_ids", []),
            "skus": values.get("skus", []),
            "tool_calls": values.get("tool_calls", []),
            "retrieved": values.get("retrieved", []),
            "steps": values.get("step_count", 0),
            "drafts": values.get("drafts_created", []),
            "guardrails": values.get("guard_flags", []),
            "usage": tracker.usage.model_dump(),
            "trace_id": trace_id,
            "prompt_version": PROMPT_VERSION,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def _waiting_answer(data: dict[str, Any], lang: Lang) -> RagAnswer:
    """A turn that stopped for the customer. `answer` is the plain text of what to confirm."""
    turn = dict(data.get("turn") or {})
    stop = {k: v for k, v in data.items() if k != "turn"}
    if not turn:
        return RagAnswer(
            answer=json.dumps(stop.get("summary", {}), ensure_ascii=False),
            language=lang,
            outcome="confirmation",
            session_id=stop.get("session_id"),
            interrupt=stop,
        )
    turn["answer"] = json.dumps(stop.get("summary", {}), ensure_ascii=False)
    turn["outcome"] = "confirmation"
    return _to_answer(turn).model_copy(update={"interrupt": stop})


def _to_answer(data: dict[str, Any]) -> RagAnswer:
    outcome = data["outcome"]
    return RagAnswer(
        answer=data["answer"],
        language=data["language"],
        route=data["route"],
        confidence=data["confidence"],
        citations=[Citation(**c) for c in data["citations"]],
        usage=data["usage"],
        order_ids=data["order_ids"],
        skus=data["skus"],
        tool_calls=data["tool_calls"],
        retrieved=[RetrievalTrace(**r) for r in data["retrieved"]],
        sufficient=outcome == "answered",
        outcome=outcome,
        steps=data["steps"],
        drafts=data.get("drafts", []),
        session_id=data["session_id"],
        latency_ms=data["latency_ms"],
        trace_id=data["trace_id"],
    )


async def _close(events: Any) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:  # pragma: no cover - cleanup only
            log.debug("closing the event stream failed", exc_info=True)
