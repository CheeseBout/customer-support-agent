"""The agent graph (SPEC 11.2, single-agent form; the multi-agent split is Phase 6).

    START -> prepare -> route -+- clarify / smalltalk / refuse ------+
                               '- agent <-> tools -- limit ----------+-> finalize -> END
                                           '-> confirm (customer) -'

`confirm` is where a request waits for the customer. It calls `interrupt()`, so the run stops
and is saved; resuming it with the customer's decision continues from the same place.

Design rules this file enforces:

* The first step of a data question must call a tool (`tool_choice="any"`), so the model cannot
  answer from memory.
* `principal` lives in the state, was written before the graph started, and no node returns it.
* Scratch messages (tool calls and results) are deleted in `finalize`. Only the question and the
  clean answer persist, so old tool data never resurfaces and history stays small.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from support_agent.agent.drafting import EDITABLE, MAX_EDITS, DraftProposer, Proposal, ProposalError
from support_agent.agent.markers import AnswerStream
from support_agent.agent.prompts import system_prompt
from support_agent.agent.state import AgentState
from support_agent.agent.tools import PROPOSE_DRAFT, ToolExecutor, schema_tools_for, wrap_result
from support_agent.core.i18n import t
from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig
from support_agent.drafts.service import DraftError, DraftService
from support_agent.llm.structured import message_text
from support_agent.memory.artifacts import comparison_markdown, request_markdown, tool_data
from support_agent.memory.summary import estimate_tokens, summarize_dialogue
from support_agent.memory.workspace import Workspace
from support_agent.rag.entities import extract_entities
from support_agent.rag.personal import wants_return_check
from support_agent.rag.retriever import Retriever
from support_agent.rag.router import route_query
from support_agent.security.guardrails import check_output
from support_agent.security.pii import redact_secrets
from support_agent.tools.client import DomainToolClient

log = logging.getLogger(__name__)

_RESET: dict[str, Any] = {
    "route": None,
    "confidence": None,
    "force_tool": False,
    "step_count": 0,
    "tool_calls": [],
    "tool_budget_used": 0,
    "docs": {},
    "retrieved": [],
    "orders_not_found": [],
    "had_data": False,
    "had_order_data": False,
    "had_refusal": False,
    "error": None,
    "raw_answer": "",
    "no_info": False,
    "cited": [],
    "answer": "",
    "outcome": None,
    "citations": [],
    "pending": None,
    "drafts_created": [],
    "guard_flags": [],
}

_ELIGIBILITY_RESULT = re.compile(
    r'source="tool:check_(?:return|warranty)_eligibility".*?"eligible":\s*(true|false)', re.DOTALL
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class AgentDeps:
    model: BaseChatModel
    retriever: Retriever
    client: DomainToolClient | None
    config: AppConfig
    # Both are needed for requests (refund, return, warranty, order); without them the agent
    # still answers questions but says it cannot submit anything.
    drafts: DraftService | None = None
    proposer: DraftProposer | None = None
    workspace: Workspace | None = None  # per-session files (comparison table, request summary)
    clock: Callable[[], datetime] = field(default=_utcnow)


# --- helpers ------------------------------------------------------------------------


def _turn_index(messages: Sequence[BaseMessage], turn_human_id: str | None) -> int:
    for i, m in enumerate(messages):
        if m.id == turn_human_id:
            return i
    return max(0, len(messages) - 1)


def _plain_dialogue(state: AgentState) -> list[BaseMessage]:
    """Every earlier question and answer, as plain text (scratch work is never kept)."""
    messages = state["messages"]
    prior = messages[: _turn_index(messages, state.get("turn_human_id"))]
    return [
        m
        for m in prior
        if isinstance(m, HumanMessage | AIMessage) and not getattr(m, "tool_calls", None)
    ]


def _prior_dialogue(state: AgentState, limit: int) -> list[BaseMessage]:
    """What the model is shown of the earlier turns.

    Without a summary: the last `limit` messages. With one: everything the summary does not
    cover yet, so nothing falls between the summary and the verbatim tail.
    """
    plain = _plain_dialogue(state)
    if state.get("summary"):
        return plain[state.get("summary_upto", 0) :]
    return plain[-limit:] if limit > 0 else []


def _preferences(state: AgentState) -> list[tuple[str, str]]:
    return [(p[0], p[1]) for p in state.get("preferences", []) if len(p) == 2]


def _model_messages(state: AgentState, limit: int) -> list[BaseMessage]:
    messages = state["messages"]
    return [
        *_prior_dialogue(state, limit),
        *messages[_turn_index(messages, state.get("turn_human_id")) :],
    ]


def _hints(state: AgentState) -> list[str]:
    """Facts our own code extracted from the message (regex), safe to state as instructions."""
    hints: list[str] = []
    if state.get("order_ids"):
        hints.append(
            f"The message mentions order id(s) {', '.join(state['order_ids'])}. Look exactly these "
            "up with the order tools; never try other ids."
        )
    if state.get("skus"):
        hints.append(f"The message mentions SKU(s) {', '.join(state['skus'])}.")
    if state.get("order_ids") and wants_return_check(state.get("question", "")):
        hints.append(
            "This is about returning or refunding: call check_return_eligibility for each order id."
        )
    return hints


def _bind(model: BaseChatModel, tools: list[BaseTool], *, force: bool) -> Any:
    if force:
        try:
            return model.bind_tools(tools, tool_choice="any")
        except (TypeError, ValueError, NotImplementedError):
            log.info("model does not support tool_choice='any'; first step is not forced")
    return model.bind_tools(tools)


def confirmation_payload(pending: dict[str, Any]) -> dict[str, Any]:
    """What the customer is shown (SPEC 14.4). Everything in it was computed by our code."""
    return {
        "id": pending["id"],
        "kind": "confirm_draft",
        "draft_type": pending["draft_type"],
        "summary": pending["summary"],
        "priority_review": pending["priority_review"],
        "allowed_decisions": ["approve", "reject", "edit"],
        "editable_fields": sorted(EDITABLE[pending["draft_type"]]),
        "expires_at": pending["expires_at"],
    }


def _no_eligible_result(scratch: Sequence[BaseMessage]) -> bool:
    """True when this turn ran eligibility checks and none of them came out eligible."""
    verdicts = [
        m.group(1)
        for msg in scratch
        if isinstance(msg, ToolMessage)
        for m in [_ELIGIBILITY_RESULT.search(message_text(msg))]
        if m
    ]
    return bool(verdicts) and "true" not in verdicts


def _known_text(messages: Sequence[BaseMessage], scratch: Sequence[BaseMessage]) -> list[str]:
    """Text this turn legitimately saw: the dialogue so far and the caller's own tool results."""
    seen = [message_text(m) for m in messages if isinstance(m, HumanMessage | AIMessage)]
    return [*seen, *(message_text(m) for m in scratch if isinstance(m, ToolMessage))]


def clean_answer(raw: str) -> tuple[str, bool, list[str]]:
    """(text without markers, no_info flag, cited ids) for a complete raw answer."""
    stream = AnswerStream()
    text = stream.feed(raw) + stream.finish()
    return text.strip(), stream.no_info, stream.cited


# --- graph --------------------------------------------------------------------------


def build_graph(deps: AgentDeps, checkpointer: Any) -> CompiledStateGraph:
    cfg = deps.config
    executor = ToolExecutor(
        client=deps.client,
        retriever=deps.retriever,
        config=cfg.agent,
        drafts=deps.drafts,
        proposer=deps.proposer,
    )

    def new_pending(proposal: Proposal, call_id: str, *, edits: int = 0) -> dict[str, Any]:
        expires = deps.clock() + timedelta(minutes=cfg.agent.confirmation_ttl_minutes)
        return {
            "id": str(uuid.uuid4()),  # the customer must answer exactly this confirmation
            "call_id": call_id,
            "draft_type": proposal.draft_type,
            "payload": proposal.payload,
            "summary": proposal.summary,
            "priority_review": proposal.priority_review,
            "args": proposal.args,
            "expires_at": expires.isoformat(),
            "edits": edits,
        }

    def save_artifact(state: AgentState, name: str, content: str | None) -> None:
        """Best-effort: a missing or unwritable workspace must never fail the turn."""
        if deps.workspace is None or not content:
            return
        try:
            deps.workspace.write(state["principal"]["user_id"], state["session_id"], name, content)
        except (OSError, ValueError):
            log.warning("could not write artifact %s", name, exc_info=True)

    def proposal_of(pending: dict[str, Any]) -> Proposal:
        return Proposal(
            pending["draft_type"],
            pending["payload"],
            pending["summary"],
            pending["priority_review"],
            pending["args"],
        )

    async def prepare(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        human = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        original = message_text(human).strip()
        text = original
        update: dict[str, Any] = dict(_RESET)
        if len(text) > cfg.guardrails.max_input_chars:
            text = text[: cfg.guardrails.max_input_chars]
        text = redact_secrets(text)  # card numbers and passwords are never stored or summarised
        if text != original:
            update["messages"] = [HumanMessage(content=text, id=human.id)]
        entities = extract_entities(text, cfg.router)
        update |= {
            "question": text,
            "turn_human_id": human.id,
            "order_ids": entities.order_ids,
            "skus": entities.skus,
        }
        return update

    async def compact(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Fold turns that left the window into the summary (SPEC FR-402).

        The transcript itself is kept (the API shows it); only what the model is shown shrinks.
        To avoid a model call every turn, it waits until `history_messages // 2` extra messages
        have piled up outside the summary, then folds everything but the last `history_messages`.
        """
        keep = cfg.agent.history_messages
        plain = _plain_dialogue(state)
        done = state.get("summary_upto", 0) if state.get("summary") else 0
        batch = max(2, keep // 2)
        if keep <= 0 or len(plain) - done <= keep + batch:
            return {}
        used = sum(estimate_tokens(message_text(m)) for m in plain[done:])
        if used + estimate_tokens(state.get("summary", "")) <= cfg.agent.summarize_after_tokens:
            return {}
        fold_to = len(plain) - keep
        try:
            summary = await summarize_dialogue(
                deps.model, state.get("summary", ""), plain[done:fold_to], state["language"], config
            )
        except Exception:  # losing a summary is better than losing the turn
            log.warning("could not summarise the conversation; keeping the plain window")
            return {}
        return {"summary": summary, "summary_upto": fold_to}

    async def route(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        decision = await route_query(
            deps.model,
            state["question"],
            cfg.router,
            history=_prior_dialogue(state, cfg.agent.history_messages),
            config=config,
        )
        get_stream_writer()(
            {"kind": "route", "data": {"route": decision.route, "language": state["language"]}}
        )
        return {
            "route": decision.route,
            "confidence": decision.confidence,
            "order_ids": decision.entities.order_ids,
            "skus": decision.entities.skus,
            "force_tool": decision.route in ("policy", "personal", "combined"),
        }

    def after_route(state: AgentState) -> str:
        if (state.get("confidence") or 0) < cfg.router.min_confidence:
            return "clarify"
        return {"chitchat": "smalltalk", "out_of_scope": "refuse"}.get(
            state["route"] or "", "agent"
        )

    def canned(message_key: str, outcome: str) -> Any:
        async def node(state: AgentState) -> dict[str, Any]:
            text = t(message_key, state["language"])  # type: ignore[arg-type]
            get_stream_writer()({"kind": "token", "data": {"text": text}})
            return {"answer": text, "outcome": outcome}

        return node

    async def agent(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        writer = get_stream_writer()
        route_name = state["route"] or "policy"
        tools = schema_tools_for(route_name, actions=executor.actions_enabled)
        messages: list[BaseMessage] = [
            SystemMessage(
                content=system_prompt(
                    state["language"],
                    _hints(state),
                    actions=executor.actions_enabled,
                    summary=state.get("summary"),
                    preferences=_preferences(state),
                )
            ),
            *_model_messages(state, cfg.agent.history_messages),
        ]
        step = state.get("step_count", 0)
        force = bool(state.get("force_tool")) and step == 0

        retries = cfg.agent.tool_retries
        ai: Any = None
        stream = AnswerStream()
        for attempt in range(retries + 1):
            stream, ai, emitted = AnswerStream(), None, False
            try:
                bound = _bind(deps.model, tools, force=force)
                async for chunk in bound.astream(messages, config):
                    ai = chunk if ai is None else ai + chunk
                    if getattr(chunk, "tool_call_chunks", None):
                        continue
                    visible = stream.feed(message_text(chunk))
                    if visible:
                        writer({"kind": "token", "data": {"text": visible}})
                        emitted = True
                break
            except Exception as exc:
                # Tokens already shown cannot be taken back, so only retry a clean failure.
                if emitted or attempt == retries:
                    raise
                log.warning("model call failed (%s); retrying", type(exc).__name__)
                force = False  # a provider that rejects tool_choice must not fail every attempt
                await asyncio.sleep(cfg.agent.tool_backoff_seconds * 2**attempt)

        tail = stream.finish()
        if tail:
            writer({"kind": "token", "data": {"text": tail}})

        if ai is None:
            return {"step_count": step + 1, "error": "EMPTY_ANSWER"}
        message = message_chunk_to_message(ai)
        if not isinstance(message, AIMessage):  # pragma: no cover - defensive
            message = AIMessage(content=message_text(ai))
        if not message.id:
            message.id = str(uuid.uuid4())
        update: dict[str, Any] = {"messages": [message], "step_count": step + 1}
        if not message.tool_calls:
            update |= {"raw_answer": stream.raw, "no_info": stream.no_info, "cited": stream.cited}
            if not stream.raw.strip():
                update["error"] = "EMPTY_ANSWER"
        return update

    def after_agent(state: AgentState) -> str:
        if state.get("error") == "EMPTY_ANSWER":
            return "finalize"
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "finalize"

    async def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        batch = await executor.run(
            [dict(call) for call in last.tool_calls],
            principal=Principal(**state["principal"]),  # type: ignore[arg-type]
            route=state["route"] or "policy",
            known_docs=state.get("docs", {}),
            budget_used=state.get("tool_budget_used", 0),
            history=state.get("tool_calls", []),
            emit=get_stream_writer(),
            language=state["language"],
        )
        missing = [o for o in batch.orders_not_found if o not in state.get("orders_not_found", [])]
        waiting = new_pending(batch.proposal[1], batch.proposal[0]) if batch.proposal else None
        return {
            "pending": waiting,
            "messages": batch.messages,
            "tool_calls": [*state.get("tool_calls", []), *batch.names],
            "tool_budget_used": state.get("tool_budget_used", 0) + len(batch.names),
            "docs": {**state.get("docs", {}), **batch.docs},
            "retrieved": [*state.get("retrieved", []), *batch.retrieved],
            "orders_not_found": [*state.get("orders_not_found", []), *missing],
            "had_data": bool(state.get("had_data")) or batch.had_data,
            "had_order_data": bool(state.get("had_order_data")) or batch.had_order_data,
            "had_refusal": bool(state.get("had_refusal")) or batch.had_refusal,
            "error": state.get("error") or batch.error,
        }

    def next_step(state: AgentState) -> str:
        if state.get("pending"):
            return "confirm"
        return "limit" if state.get("step_count", 0) >= cfg.agent.max_steps else "agent"

    async def confirm(state: AgentState) -> dict[str, Any]:
        """Stop for the customer, then act on their decision.

        The node runs again from the top when resumed, so everything before `interrupt()` must
        be free of side effects. Creating the draft comes after it, and is idempotent.
        """
        pending = state["pending"]
        assert pending is not None and deps.drafts is not None and deps.proposer is not None
        reply = interrupt(confirmation_payload(pending))
        reply = reply if isinstance(reply, dict) else {}
        principal = Principal(**state["principal"])  # type: ignore[arg-type]

        def finish(
            outcome: dict[str, Any], created: dict[str, Any] | None = None
        ) -> dict[str, Any]:
            message = ToolMessage(
                content=wrap_result(PROPOSE_DRAFT, json.dumps(outcome, ensure_ascii=False)),
                tool_call_id=pending["call_id"],
                name=PROPOSE_DRAFT,
            )
            update: dict[str, Any] = {"messages": [message], "pending": None, "had_data": True}
            if created is not None:
                update["drafts_created"] = [*state.get("drafts_created", []), created]
            return update

        decision = reply.get("decision")
        if reply.get("interrupt_id") != pending["id"]:
            return finish({"status": "error", "reason": "this confirmation is no longer current"})
        if deps.clock() > datetime.fromisoformat(pending["expires_at"]):
            return finish({"status": "expired", "reason": "the confirmation timed out"})
        if decision == "reject":
            return finish({"status": "declined_by_customer"})
        if decision == "approve":
            try:
                draft, created = await deps.drafts.create(
                    principal, pending["draft_type"], pending["payload"], state["session_id"]
                )
            except DraftError as exc:
                return finish({"status": "error", "code": exc.code, "message": exc.message})
            info = {
                "id": draft.id,
                "type": draft.type,
                "status": draft.status,
                "priority_review": draft.priority_review,
            }
            save_artifact(
                state,
                "request_summary.md",
                request_markdown(info, pending["summary"], state["language"]),
            )
            outcome = {
                "status": "created",
                "review": "waiting for staff",
                "id": draft.id,
                "type": draft.type,
                "draft_status": draft.status,
                "priority_review": draft.priority_review,
            }
            return finish(
                outcome | ({} if created else {"note": "already submitted earlier"}), info
            )
        if decision == "edit":
            if pending["edits"] >= MAX_EDITS:
                return finish({"status": "too_many_edits"})
            try:
                revised = await deps.proposer.edited(
                    proposal_of(pending), reply.get("edits") or {}, principal
                )
            except ProposalError as exc:
                return finish(
                    {"status": "edit_rejected", "code": exc.code, "message": exc.message}
                    | exc.details
                )
            # Ask again with the corrected request: a new confirmation, a new expiry.
            return {"pending": new_pending(revised, pending["call_id"], edits=pending["edits"] + 1)}
        return finish({"status": "error", "reason": "unknown decision"})

    async def limit(state: AgentState) -> dict[str, Any]:
        return {"error": "STEP_LIMIT"}

    async def finalize(state: AgentState) -> dict[str, Any]:
        messages = state["messages"]
        scratch = messages[_turn_index(messages, state.get("turn_human_id")) + 1 :]
        final_ai = next(
            (m for m in reversed(scratch) if isinstance(m, AIMessage) and not m.tool_calls), None
        )
        lang = state["language"]
        route_name = state.get("route")
        error = state.get("error")
        had_data = bool(state.get("had_data"))
        docs = state.get("docs", {})
        writer = get_stream_writer()

        citations: list[dict[str, str]] = []
        model_text, _, _ = clean_answer(state.get("raw_answer", ""))
        streamed = bool(model_text)  # the customer has already seen the model's words
        from_model = False
        if state.get("outcome") in ("clarify", "refused"):
            text, outcome = state["answer"], state["outcome"]
        elif error == "STEP_LIMIT":
            text, outcome = t("step_limit", lang), "error"  # type: ignore[arg-type]
        elif error == "EMPTY_ANSWER":
            text, outcome = t("upstream_error", lang), "error"  # type: ignore[arg-type]
        elif error in ("UPSTREAM_ERROR", "TIMEOUT") and not had_data:
            text, outcome = model_text or t("upstream_error", lang), "error"  # type: ignore[arg-type]
        else:
            text = model_text
            from_model = True
            # An order that does not exist stays "not_found" even if the model went on to call
            # unrelated tools (a catalogue search): only order data or policy text changes that.
            if state.get("orders_not_found") and not (state.get("had_order_data") or docs):
                outcome = "not_found"
            elif (state.get("no_info") and not state.get("had_refusal")) or not had_data:
                # Either the model said so (a "no" from the shop's own rules is an answer, so the
                # marker is ignored then), or nothing it was told came from real data.
                outcome = "no_info"
            else:
                outcome = "answered"
                cited = [c for c in state.get("cited", []) if c in docs]
                if not cited and docs and route_name in ("policy", "combined"):
                    cited = [next(iter(docs))]  # a policy answer always carries a source (FR-002)
                citations = [
                    {"source": docs[c]["doc_id"], "section": docs[c]["section"]} for c in cited
                ]
        guard_flags: list[str] = []
        if from_model:  # only the model's own words need screening
            verdict = check_output(
                text,
                cfg=cfg.guardrails,
                known_text=_known_text(messages[: len(messages) - len(scratch)], scratch),
                system_prompt=system_prompt(lang),
                ineligible=_no_eligible_result(scratch),
                safe_leak=t("internal_hidden", lang),  # type: ignore[arg-type]
                safe_promise=t("no_promise", lang),  # type: ignore[arg-type]
            )
            if verdict.changed:
                guard_flags = verdict.flags
                log.warning("output guardrail changed the answer", extra={"flags": guard_flags})
                text = verdict.text
                if guard_flags != ["pii_masked"]:  # a withdrawn answer keeps no sources
                    citations = []
                if "prompt_or_secret_leak" in guard_flags:
                    outcome = "refused"
                if streamed:
                    writer({"kind": "replace", "data": {"text": text}})
                else:
                    writer({"kind": "token", "data": {"text": text}})
        if outcome == "error" and not streamed:
            writer({"kind": "token", "data": {"text": text}})
        for c in citations:
            writer({"kind": "citation", "data": c})

        compared = next(
            (
                m
                for m in reversed(scratch)
                if isinstance(m, ToolMessage) and m.name == "compare_products"
            ),
            None,
        )
        if compared is not None:
            data = tool_data(message_text(compared))
            save_artifact(
                state, "comparison_table.md", comparison_markdown(data, lang) if data else None
            )

        final_id = final_ai.id if final_ai is not None and final_ai.id else str(uuid.uuid4())
        keep = {final_id}
        update_messages: list[BaseMessage] = [
            RemoveMessage(id=m.id) for m in scratch if m.id and m.id not in keep
        ]
        update_messages.append(AIMessage(content=text, id=final_id))
        return {
            "messages": update_messages,
            "answer": text,
            "outcome": outcome,
            "citations": citations,
            "guard_flags": guard_flags,
        }

    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare)
    graph.add_node("compact", compact)
    graph.add_node("route", route)
    graph.add_node("clarify", canned("clarify", "clarify"))
    graph.add_node("smalltalk", canned("chitchat", "refused"))
    graph.add_node("refuse", canned("out_of_scope", "refused"))
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_node("confirm", confirm)
    graph.add_node("limit", limit)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "compact")
    graph.add_edge("compact", "route")
    graph.add_conditional_edges(
        "route",
        after_route,
        {"clarify": "clarify", "smalltalk": "smalltalk", "refuse": "refuse", "agent": "agent"},
    )
    for canned_node in ("clarify", "smalltalk", "refuse"):
        graph.add_edge(canned_node, "finalize")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "finalize": "finalize"})
    awaiting: dict[Hashable, str] = {"agent": "agent", "limit": "limit", "confirm": "confirm"}
    graph.add_conditional_edges("tools", next_step, awaiting)
    graph.add_conditional_edges("confirm", next_step, awaiting)
    graph.add_edge("limit", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
