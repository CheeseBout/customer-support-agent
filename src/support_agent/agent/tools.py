"""The agent's tool node: policy search, the MCP domain tools, drafts and skills.

This is the security choke point of the loop. The model proposes calls; this module decides
whether they run:

* the identity comes from the run state, never from the model;
* a tool outside the route's allow-list is refused (least privilege);
* arguments are validated against a closed schema, so `customer_id` and friends fail;
* every call is budgeted, retried with backoff on infrastructure errors, and its result is
  wrapped as untrusted data before the model sees it;
* `propose_draft` never writes anything: it returns a validated proposal that the graph puts
  in front of the customer, and only their confirmation creates a draft.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ValidationError

from support_agent.agent.drafting import DraftProposer, Proposal, ProposalError, ProposeDraftArgs
from support_agent.agent.skills import load_skill, skill_facts, skill_names
from support_agent.core.principal import Principal
from support_agent.core.results import ToolError, ToolResult
from support_agent.core.retry import with_backoff
from support_agent.core.settings import AgentConfig, BusinessRules
from support_agent.drafts.models import Draft
from support_agent.drafts.service import DraftError, DraftService
from support_agent.mcp_db.server import IDENTITY_ARGS
from support_agent.rag.answer import neutralise
from support_agent.rag.retriever import RetrievedChunk, Retriever
from support_agent.security.pii import mask_text
from support_agent.tools.client import DomainToolClient
from support_agent.tools.langchain_tools import (
    ARGS_SCHEMAS,
    DOMAIN_TOOL_NAMES,
    TOOL_SPECS,
    ToolArgs,
    schema_tool,
)

log = logging.getLogger(__name__)

SEARCH_POLICY = "search_policy"
PROPOSE_DRAFT = "propose_draft"
LIST_MY_DRAFTS = "list_my_drafts"
CANCEL_DRAFT = "cancel_draft"
LOAD_SKILL = "load_skill"
ACTION_TOOLS = frozenset({PROPOSE_DRAFT, LIST_MY_DRAFTS, CANCEL_DRAFT})
ORDER_TOOLS = frozenset(
    {"get_order", "get_shipment_status", "check_return_eligibility", "check_warranty_eligibility"}
)
# Results computed by our rules engine, not read from free text.
AUTHORITATIVE_TOOLS = frozenset(
    {"check_return_eligibility", "check_warranty_eligibility", PROPOSE_DRAFT}
)
RETRYABLE = ("UPSTREAM_ERROR", "TIMEOUT")
# A "no" that the shop's own rules or stock produced. It is a real answer to the customer's
# request (unlike a malformed call), so it counts as data when the outcome is labelled.
BUSINESS_REFUSALS = frozenset({"NOT_ELIGIBLE", "OUT_OF_STOCK", "LIMIT_EXCEEDED", "NOT_SUPPORTED"})
# Lookups may not starve the request the customer asked for: propose_draft keeps a small reserve
# beyond `max_tool_calls`, and a tool that models tend to repeat is capped per question.
ACTION_RESERVE = 2
PER_TOOL_LIMIT = {"search_products": 3}


class SearchPolicyArgs(ToolArgs):
    query: str = Field(min_length=2, description="What to look up, in the customer's wording")


class ListDraftsArgs(ToolArgs):
    status: Literal["pending", "approved", "rejected", "cancelled"] | None = None


class CancelDraftArgs(ToolArgs):
    draft_id: str = Field(min_length=3, description="The id of one of the customer's own requests")


class LoadSkillArgs(ToolArgs):
    name: str = Field(description="Skill name, e.g. return-item")


INTERNAL_SPECS: list[tuple[str, str, type[BaseModel]]] = [
    (
        SEARCH_POLICY,
        "Search the shop's policy documents (returns, refunds, exchanges, shipping, warranty, "
        "payment). Returns numbered documents to cite.",
        SearchPolicyArgs,
    ),
    (
        PROPOSE_DRAFT,
        "Ask the customer to confirm a refund, return, warranty claim, new order or a request "
        "to be passed to a person. Give only "
        "the type, order, item, reason or order details: amounts and eligibility are worked out "
        "by the system. Nothing is submitted until the customer confirms.",
        ProposeDraftArgs,
    ),
    (
        LIST_MY_DRAFTS,
        "List the customer's own requests (refund, return, warranty, order) and their status.",
        ListDraftsArgs,
    ),
    (
        CANCEL_DRAFT,
        "Withdraw one of the customer's own requests that is still pending.",
        CancelDraftArgs,
    ),
    (
        LOAD_SKILL,
        "Load a step-by-step procedure for a common task (see the skill list in the prompt).",
        LoadSkillArgs,
    ),
]
POLICY_SPEC = INTERNAL_SPECS[0]
_ALL_ARGS: dict[str, type[BaseModel]] = {
    **ARGS_SCHEMAS,
    **{name: schema for name, _, schema in INTERNAL_SPECS},
}

# Which tools each route may use (SPEC 13.3: allow-list per route).
ROUTE_TOOLS: dict[str, frozenset[str]] = {
    "policy": frozenset({SEARCH_POLICY, LOAD_SKILL}),
    "personal": DOMAIN_TOOL_NAMES | ACTION_TOOLS | {LOAD_SKILL},
    "combined": DOMAIN_TOOL_NAMES | ACTION_TOOLS | {SEARCH_POLICY, LOAD_SKILL},
}


def allowed_tools(
    route: str, *, actions: bool = True, domain: frozenset[str] | None = None
) -> frozenset[str]:
    """Tools for `route`. `domain` lists the data tools this shop has (None: all of them)."""
    names = ROUTE_TOOLS.get(route, frozenset())
    if domain is not None:
        names = names - (DOMAIN_TOOL_NAMES - domain)
    return names if actions else names - ACTION_TOOLS


def schema_tools_for(
    route: str, *, actions: bool = True, domain: frozenset[str] | None = None
) -> list[BaseTool]:
    """Self-describing (non-executable) tools the model may see for `route`."""
    allowed = allowed_tools(route, actions=actions, domain=domain)
    return [schema_tool(n, d, s) for n, d, s in [*INTERNAL_SPECS, *TOOL_SPECS] if n in allowed]


@dataclass
class ToolBatch:
    """What one round of tool calls changed."""

    messages: list[ToolMessage] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    docs: dict[str, dict[str, Any]] = field(default_factory=dict)  # newly registered documents
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    orders_not_found: list[str] = field(default_factory=list)
    had_data: bool = False
    had_order_data: bool = False  # an order tool returned the customer's own order data
    had_refusal: bool = False  # the shop's rules or stock said no (a complete answer)
    error: str | None = None  # an infrastructure failure survived all retries
    # A validated request waiting for the customer: (tool call id, proposal). Its tool result is
    # produced after they decide, so no ToolMessage exists for it yet.
    proposal: tuple[str, Proposal] | None = None


@dataclass
class _Outcome:
    result: ToolResult
    chunks: list[RetrievedChunk] = field(default_factory=list)
    proposal: Proposal | None = None
    trusted: str | None = None  # shop-authored text (a skill) that needs no untrusted wrapper


def _summary(args: Any) -> str:
    return mask_text(json.dumps(args, ensure_ascii=False, default=str))[:200]


def wrap_result(name: str, payload: str) -> str:
    """Tool output as the model sees it: always inside a delimiter, never as bare text."""
    payload = neutralise(payload)
    if name in AUTHORITATIVE_TOOLS:
        return f'<fact source="tool:{name}" authoritative="true">{payload}</fact>'
    return f'<untrusted_data source="tool:{name}">{payload}</untrusted_data>'


def draft_view(draft: Draft) -> dict[str, Any]:
    """A draft as told to the customer: what they asked for and where it stands."""
    view: dict[str, Any] = {
        "id": draft.id,
        "type": draft.type,
        "status": draft.status,
        "created_at": draft.created_at.isoformat(),
        "priority_review": draft.priority_review,
    }
    payload = draft.payload
    if draft.type in ("refund", "return"):
        view |= {
            "order_id": payload.get("order_id"),
            "items": payload.get("items"),
            "amount": payload.get("refundable_amount"),
        }
    elif draft.type == "warranty":
        view |= {"order_id": payload.get("order_id"), "sku": payload.get("sku")}
    elif draft.type == "handoff":
        view |= {"order_id": payload.get("order_id"), "reason": payload.get("reason")}
    else:
        view |= {"total": payload.get("total"), "currency": payload.get("currency")}
    if draft.status in ("approved", "rejected") and draft.review_note:
        view["staff_note"] = draft.review_note
    return view


class ToolExecutor:
    def __init__(
        self,
        *,
        client: DomainToolClient | None,
        retriever: Retriever,
        config: AgentConfig,
        drafts: DraftService | None = None,
        proposer: DraftProposer | None = None,
        rules: BusinessRules | None = None,
        domain_tools: frozenset[str] | None = None,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.cfg = config
        self.drafts = drafts
        self.proposer = proposer
        self.rules = rules or BusinessRules()
        self.domain_tools = domain_tools  # the data tools this shop has; None means all

    @property
    def actions_enabled(self) -> bool:
        """Requests need somewhere to store drafts, a way to check eligibility first, and at
        least one kind of request the shop takes through the assistant."""
        return (
            self.drafts is not None
            and self.proposer is not None
            and bool(self.proposer.request_types)
        )

    async def run(
        self,
        calls: list[dict[str, Any]],
        *,
        principal: Principal,
        route: str,
        known_docs: dict[str, dict[str, Any]],
        budget_used: int,
        emit: Callable[[dict[str, Any]], None],
        language: str = "en",
        history: Sequence[str] = (),
    ) -> ToolBatch:
        batch = ToolBatch()
        registry = dict(known_docs)  # shared by this round's calls so ids never collide
        allowed = allowed_tools(route, actions=self.actions_enabled, domain=self.domain_tools)

        async def one(index: int, call: dict[str, Any]) -> tuple[str, str, _Outcome]:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            call_id = str(call.get("id") or f"call_{index}")
            emit(
                {
                    "kind": "tool_start",
                    "data": {"call_id": call_id, "tool": name, "args_summary": _summary(args)},
                }
            )
            started = time.perf_counter()
            limit = self.cfg.max_tool_calls + (ACTION_RESERVE if name == PROPOSE_DRAFT else 0)
            repeats = list(history).count(name) + sum(
                1 for earlier in calls[:index] if earlier.get("name") == name
            )
            if budget_used + index >= limit:
                outcome = _Outcome(
                    ToolResult.failure(
                        "FORBIDDEN",
                        "Tool budget for this question is used up; answer with what you have.",
                    )
                )
            elif name in PER_TOOL_LIMIT and repeats >= PER_TOOL_LIMIT[name]:
                outcome = _Outcome(
                    ToolResult.failure(
                        "FORBIDDEN",
                        f"You already called {name} {repeats} times for this question; "
                        "use the results you have.",
                    )
                )
            elif name not in allowed:
                outcome = _Outcome(
                    ToolResult.failure(
                        "FORBIDDEN", f"Tool {name!r} is not available for this question."
                    )
                )
            else:
                outcome = await self._execute(name, args, principal, language)
            emit(
                {
                    "kind": "tool_end",
                    "data": {
                        "call_id": call_id,
                        "tool": name,
                        "ok": outcome.result.ok,
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                    },
                }
            )
            return call_id, name, outcome

        outcomes = await asyncio.gather(*(one(i, c) for i, c in enumerate(calls)))

        for (call_id, name, outcome), call in zip(outcomes, calls, strict=True):
            batch.names.append(name)
            result = outcome.result
            if outcome.proposal is not None:
                if batch.proposal is None:
                    batch.proposal = (call_id, outcome.proposal)
                    batch.had_data = batch.had_order_data = True  # built from the shop's own data
                    continue  # answered once the customer has decided
                result = ToolResult.failure(
                    "CONFLICT",
                    "Only one request can be confirmed at a time; propose the next later.",
                )
            if outcome.trusted is not None:
                content = outcome.trusted
                # A procedure is not an answer to the question: it must not turn a missing order
                # or an empty search into an "answered" outcome.
                batch.had_data = batch.had_data or name != LOAD_SKILL
            elif name == SEARCH_POLICY and result.ok:
                content = self._render_documents(outcome.chunks, registry, batch)
                batch.had_data = batch.had_data or bool(outcome.chunks)
            else:
                content = wrap_result(name, json.dumps(result.to_wire(), ensure_ascii=False))
                refused = bool(
                    result.error
                    and (
                        result.error.code in BUSINESS_REFUSALS
                        or (name == CANCEL_DRAFT and result.error.code == "NOT_FOUND")
                    )
                )
                batch.had_refusal = batch.had_refusal or refused
                batch.had_data = batch.had_data or result.ok or refused
                if name in ORDER_TOOLS and (result.ok or refused):
                    batch.had_order_data = True
            batch.messages.append(ToolMessage(content=content, tool_call_id=call_id, name=name))
            if result.error:
                if result.error.code == "NOT_FOUND" and (
                    name in ORDER_TOOLS or name == PROPOSE_DRAFT
                ):
                    order_id = str((call.get("args") or {}).get("order_id", ""))
                    if order_id and order_id not in batch.orders_not_found:
                        batch.orders_not_found.append(order_id)
                if result.error.code in RETRYABLE:
                    batch.error = batch.error or result.error.code
        return batch

    # --- one call --------------------------------------------------------------------
    async def _execute(
        self, name: str, args: dict[str, Any], principal: Principal, language: str
    ) -> _Outcome:
        schema = _ALL_ARGS.get(name)
        if schema is None:
            return _Outcome(ToolResult.failure("INVALID_ARGUMENT", f"Unknown tool {name!r}."))
        identity = set(args) & IDENTITY_ARGS
        if identity:  # a prompt-injection signature: refuse loudly
            log.warning("tool %s: identity argument(s) refused: %s", name, sorted(identity))
            return _Outcome(
                ToolResult.failure("FORBIDDEN", "Identity cannot be supplied as a tool argument.")
            )
        try:
            parsed = schema.model_validate(args)
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) or "?" for e in exc.errors()})
            return _Outcome(ToolResult.failure("INVALID_ARGUMENT", f"Invalid arguments: {fields}."))

        if isinstance(parsed, SearchPolicyArgs):
            result, chunks = await self._search_policy(parsed.query)
            return _Outcome(result, chunks)
        if isinstance(parsed, ProposeDraftArgs):
            return await self._propose(parsed, principal)
        if isinstance(parsed, ListDraftsArgs):
            return await self._list_drafts(parsed, principal)
        if isinstance(parsed, CancelDraftArgs):
            return await self._cancel_draft(parsed, principal)
        if isinstance(parsed, LoadSkillArgs):
            return self._skill(parsed.name, language)

        if self.client is None:
            return _Outcome(
                ToolResult.failure("UPSTREAM_ERROR", "The data service is not available.")
            )
        payload = parsed.model_dump(mode="json", exclude_none=True)
        client = self.client
        result = await with_backoff(
            lambda: client.call(name, payload, principal),
            retries=self.cfg.tool_retries,
            base_delay=self.cfg.tool_backoff_seconds,
            retry_if=lambda r: bool(r.error and r.error.code in RETRYABLE),
        )
        return _Outcome(result)

    async def _search_policy(self, query: str) -> tuple[ToolResult, list[RetrievedChunk]]:
        try:
            chunks = await with_backoff(
                lambda: self.retriever.retrieve(query),
                retries=self.cfg.tool_retries,
                base_delay=self.cfg.tool_backoff_seconds,
            )
        except Exception as exc:
            log.warning("policy search failed: %s", type(exc).__name__)
            return ToolResult.failure("UPSTREAM_ERROR", "The policy search is unavailable."), []
        return ToolResult.success({"documents": len(chunks)}), chunks

    # --- requests (drafts) ---------------------------------------------------------------------
    async def _propose(self, args: ProposeDraftArgs, principal: Principal) -> _Outcome:
        assert self.proposer is not None  # guaranteed by the allow-list
        try:
            proposal = await self.proposer.propose(args, principal)
        except ProposalError as exc:
            failure = ToolResult.failure(exc.code, exc.message)
            failure.data = exc.details or None
            return _Outcome(failure)
        except Exception:
            log.exception("could not prepare the request")
            return _Outcome(ToolResult.failure("UPSTREAM_ERROR", "Could not prepare the request."))
        return _Outcome(
            ToolResult.success({"status": "awaiting_customer_confirmation"}), proposal=proposal
        )

    async def _list_drafts(self, args: ListDraftsArgs, principal: Principal) -> _Outcome:
        assert self.drafts is not None
        found = await self.drafts.list_for(
            principal, statuses=[args.status] if args.status else None, limit=10
        )
        return _Outcome(ToolResult.success({"requests": [draft_view(d) for d in found]}))

    async def _cancel_draft(self, args: CancelDraftArgs, principal: Principal) -> _Outcome:
        assert self.drafts is not None
        try:
            draft = await self.drafts.cancel(principal, args.draft_id)
        except DraftError as exc:
            return _Outcome(
                ToolResult(ok=False, error=ToolError(code=exc.code, message=exc.message))
            )
        return _Outcome(ToolResult.success({"id": draft.id, "status": draft.status}))

    def _skill(self, name: str, language: str) -> _Outcome:
        skill = load_skill(name.strip(), language, skill_facts(self.rules, language))
        if skill is None:
            return _Outcome(
                ToolResult.failure(
                    "INVALID_ARGUMENT", f"No such skill {name!r}. Available: {skill_names()}."
                )
            )
        text = f'<skill name="{skill.name}" lang="{skill.lang}">\n{skill.body}\n</skill>'
        return _Outcome(ToolResult.success({"skill": skill.name}), trusted=text)

    # --- rendering -------------------------------------------------------------------
    @staticmethod
    def _render_documents(
        chunks: list[RetrievedChunk], registry: dict[str, dict[str, Any]], batch: ToolBatch
    ) -> str:
        """Numbered, delimited documents. The same chunk keeps its id across calls."""
        if not chunks:
            return "<documents></documents>\nNo policy text is relevant to this query."
        parts = []
        for chunk in chunks:
            doc_id = next(
                (
                    k
                    for k, v in registry.items()
                    if v["doc_id"] == chunk.doc_id and v["section"] == chunk.section
                ),
                None,
            )
            if doc_id is None:
                doc_id = f"D{max((int(k[1:]) for k in registry), default=0) + 1}"
                info = {
                    "doc_id": chunk.doc_id,
                    "section": chunk.section,
                    "source": chunk.source,
                    "score": chunk.score,
                }
                registry[doc_id] = info
                batch.docs[doc_id] = info
                batch.retrieved.append(
                    {"doc_id": chunk.doc_id, "section": chunk.section, "score": chunk.score}
                )
            parts.append(
                f'<document id="{doc_id}" source="{chunk.doc_id}" '
                f'section="{neutralise(chunk.section)}">\n{neutralise(chunk.text)}\n</document>'
            )
        return "<documents>\n" + "\n".join(parts) + "\n</documents>"
