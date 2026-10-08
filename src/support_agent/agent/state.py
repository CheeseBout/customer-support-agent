"""Agent state (SPEC 11.1).

Everything is plain data (dicts, lists, strings) so any checkpointer can serialise it without
custom type registration. Per-turn fields are reset by the `prepare` node; only `messages`
survive from turn to turn.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # --- conversation (persisted across turns) ----------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]

    # --- memory (persisted across turns) -----------------------------------------------
    summary: str  # earlier turns, folded by the `compact` node
    summary_upto: int  # how many earlier plain messages the summary already covers

    # --- identity: written once, by SupportAgent, before the graph starts ------------
    principal: dict[str, str]  # {"user_id": ..., "role": ...}. No node may change it.
    session_id: str
    language: str  # "vi" | "en"
    preferences: list[list[str]]  # saved [key, value] pairs, supplied per turn (data, not orders)

    # --- per turn --------------------------------------------------------------------
    question: str
    turn_human_id: str  # id of this turn's HumanMessage: everything after it is scratch work
    route: str | None
    confidence: float | None
    order_ids: list[str]
    skus: list[str]
    force_tool: bool  # first step must call a tool (the answer must come from data)
    step_count: int
    tool_calls: list[str]  # names, in call order
    tool_budget_used: int
    docs: dict[str, dict[str, Any]]  # "D1" -> {doc_id, section, source, score}
    retrieved: list[dict[str, Any]]
    orders_not_found: list[str]
    had_data: bool  # at least one tool returned real data
    had_order_data: bool  # an order tool returned the customer's own order (or a rules verdict)
    had_refusal: bool  # the rules or stock said no: an answer even if the model says [NO_INFO]
    error: str | None  # UPSTREAM_ERROR | TIMEOUT | STEP_LIMIT | EMPTY_ANSWER
    raw_answer: str  # the final model text, markers included
    no_info: bool
    cited: list[str]  # ["D1", ...] in order of first citation
    guard_flags: list[str]  # what the guardrails did this turn: pii_masked, false_commitment, ...

    # --- requests awaiting the customer's confirmation (SPEC 10.5) ---------------------
    pending: dict[str, Any] | None  # the proposal being confirmed: id, payload, summary, expiry
    drafts_created: list[dict[str, Any]]  # drafts made this turn: id, type, status

    # --- result ----------------------------------------------------------------------
    answer: str
    outcome: str | None
    citations: list[dict[str, str]]
