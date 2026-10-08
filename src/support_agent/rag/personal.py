"""Gather the customer's own data through the domain tools.

Step 1 is deterministic: order ids / SKUs found by regex are looked up exactly (FR-007).
Step 2 lets the LLM pick tools only when nothing was extracted ("where is my order?").
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from support_agent.core.principal import Principal
from support_agent.core.results import ToolResult
from support_agent.mcp_db.service import fold
from support_agent.rag.answer import Fact
from support_agent.rag.entities import Entities
from support_agent.tools.client import DomainToolClient
from support_agent.tools.langchain_tools import build_domain_tools

log = logging.getLogger(__name__)

MAX_PREFETCH_ORDERS = 3
MAX_PREFETCH_SKUS = 3

# Matched on accent-folded text. Vietnamese "trả" folds to "tra", which also starts "tra cứu"
# (look up), so that reading is excluded.
_RETURN_INTENT = re.compile(
    r"\b(return|refund|exchange|hoan tien|doi hang|doi tra|doi san pham)\b"
    r"|\btra\b(?!\s*(cuu|xet|loi|gop))"
)

TOOL_PROMPT = """\
You gather facts about the customer's own orders and products by calling tools. Do NOT answer \
the customer yourself. Call whichever tools are needed to find the facts that the question \
requires, then stop calling tools. Identity is handled for you: never pass a customer id. \
Order ids are plain ids without a leading '#'.
"""


@dataclass
class PersonalResult:
    facts: list[Fact] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    orders_not_found: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def has_data(self) -> bool:
        return any(f.data.get("ok") for f in self.facts if isinstance(f.data, dict))


def wants_return_check(question: str) -> bool:
    return bool(_RETURN_INTENT.search(fold(question)))


def _label(name: str, args: dict[str, Any]) -> str:
    return f"{name}({', '.join(f'{k}={v}' for k, v in args.items())})"


async def _call(
    client: DomainToolClient,
    principal: Principal,
    result: PersonalResult,
    name: str,
    args: dict[str, Any],
    *,
    authoritative: bool = False,
) -> ToolResult:
    r = await client.call(name, args, principal)
    result.tool_calls.append(name)
    result.facts.append(
        Fact(label=_label(name, args), data=r.to_wire(), authoritative=authoritative)
    )
    if r.error and r.error.code == "UPSTREAM_ERROR":
        result.error = "UPSTREAM_ERROR"
    elif r.error and r.error.code == "TIMEOUT":
        result.error = "TIMEOUT"
    return r


async def prefetch(
    client: DomainToolClient, principal: Principal, question: str, entities: Entities
) -> PersonalResult:
    result = PersonalResult()
    return_intent = wants_return_check(question)
    for order_id in entities.order_ids[:MAX_PREFETCH_ORDERS]:
        order = await _call(client, principal, result, "get_order", {"order_id": order_id})
        if not order.ok:
            if order.error and order.error.code == "NOT_FOUND":
                result.orders_not_found.append(order_id)
            continue
        await _call(client, principal, result, "get_shipment_status", {"order_id": order_id})
        if return_intent:
            await _call(
                client,
                principal,
                result,
                "check_return_eligibility",
                {"order_id": order_id},
                authoritative=True,
            )
    for sku in entities.skus[:MAX_PREFETCH_SKUS]:
        await _call(client, principal, result, "check_stock", {"sku": sku})
    return result


async def tool_loop(
    model: BaseChatModel,
    tools: list[BaseTool],
    question: str,
    result: PersonalResult,
    *,
    history: list[BaseMessage] | None = None,
    max_steps: int = 4,
    config: RunnableConfig | None = None,
) -> None:
    """Let the model call tools (bounded). Facts are appended to `result`."""
    try:
        bound = model.bind_tools(tools)
    except NotImplementedError:
        log.warning("model has no tool calling; skipping tool loop")
        return
    by_name = {t.name: t for t in tools}
    messages: list[BaseMessage] = [
        SystemMessage(content=TOOL_PROMPT),
        *(history or []),
        HumanMessage(content=question),
    ]
    for _ in range(max_steps):
        ai = await bound.ainvoke(messages, config=config)
        calls = getattr(ai, "tool_calls", None) or []
        if not calls:
            return
        messages.append(ai)
        for call in calls:
            name, args = call["name"], call.get("args", {})
            tool = by_name.get(name)
            if tool is None:
                content = json.dumps(
                    ToolResult.failure("INVALID_ARGUMENT", f"Unknown tool {name}").to_wire()
                )
            else:
                try:
                    content = await tool.ainvoke(args)
                except Exception as exc:  # pydantic rejected the args (e.g. customer_id)
                    log.warning("tool %s rejected arguments: %s", name, type(exc).__name__)
                    content = json.dumps(
                        ToolResult.failure("INVALID_ARGUMENT", "Invalid tool arguments.").to_wire()
                    )
            result.tool_calls.append(name)
            try:
                data = json.loads(content)
            except ValueError:
                data = {"ok": False}
            result.facts.append(Fact(label=_label(name, args), data=data))
            err = (data.get("error") or {}).get("code") if isinstance(data, dict) else None
            if err in ("UPSTREAM_ERROR", "TIMEOUT"):
                result.error = err
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))


async def gather_personal(
    client: DomainToolClient,
    principal: Principal,
    model: BaseChatModel | None,
    question: str,
    entities: Entities,
    *,
    history: list[BaseMessage] | None = None,
    max_steps: int = 4,
    config: RunnableConfig | None = None,
) -> PersonalResult:
    result = await prefetch(client, principal, question, entities)
    if entities.empty and model is not None:
        await tool_loop(
            model,
            build_domain_tools(client, principal),
            question,
            result,
            history=history,
            max_steps=max_steps,
            config=config,
        )
    return result
