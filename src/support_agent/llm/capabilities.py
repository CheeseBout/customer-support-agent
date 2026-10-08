"""Startup capability check: plain call, tool calling, structured output (SPEC 6.2)."""

from __future__ import annotations

import logging
import time
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from support_agent.llm.structured import message_text

log = logging.getLogger(__name__)

Status = Literal["ok", "degraded", "failed"]


class CapabilityReport(BaseModel):
    status: Status
    basic: bool = False
    tool_calling: bool = False
    structured_output: bool = False
    latency_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


class _Probe(BaseModel):
    city: str
    temperature_c: int


@tool
def get_probe_temperature(city: str) -> str:
    """Return the temperature of a city. Always call this tool when asked about temperature."""
    return f"{city}: 21C"


async def check_capabilities(model: BaseChatModel) -> CapabilityReport:
    """Probe `model`. `failed` = unusable, `degraded` = missing tool calling/structured output."""
    report = CapabilityReport(status="failed")
    started = time.perf_counter()

    try:
        reply = await model.ainvoke([HumanMessage(content="Reply with the single word: pong")])
        report.basic = bool(message_text(reply).strip())
    except Exception as exc:
        report.warnings.append(f"basic call failed: {type(exc).__name__}: {exc}")
        report.latency_ms = int((time.perf_counter() - started) * 1000)
        return report
    report.latency_ms = int((time.perf_counter() - started) * 1000)
    if not report.basic:
        report.warnings.append("basic call returned an empty reply")
        return report

    try:
        with_tools = model.bind_tools([get_probe_temperature])
        reply = await with_tools.ainvoke(
            [HumanMessage(content="What is the temperature in Hanoi? Use the tool.")]
        )
        calls = getattr(reply, "tool_calls", None) or []
        report.tool_calling = any(c.get("name") == "get_probe_temperature" for c in calls)
        if not report.tool_calling:
            report.warnings.append("model did not emit the expected tool call")
    except Exception as exc:
        report.warnings.append(f"tool calling unsupported: {type(exc).__name__}: {exc}")

    try:
        parsed = await model.with_structured_output(_Probe).ainvoke(
            [HumanMessage(content="It is 21 degrees Celsius in Hanoi right now.")]
        )
        if isinstance(parsed, dict):
            parsed = _Probe.model_validate(parsed)
        report.structured_output = isinstance(parsed, _Probe) and parsed.temperature_c == 21
        if not report.structured_output:
            report.warnings.append("structured output returned unexpected content")
    except Exception as exc:
        report.warnings.append(f"structured output unsupported: {type(exc).__name__}: {exc}")

    report.status = "ok" if report.tool_calling and report.structured_output else "degraded"
    if report.status == "degraded":
        log.warning("model capability degraded: %s", "; ".join(report.warnings))
    return report


class CapabilityFailure(RuntimeError):
    pass


def enforce(report: CapabilityReport, *, strict: bool) -> None:
    """Raise when `strict` and the model is not fully capable (used by `serve`)."""
    if strict and report.status != "ok":
        raise CapabilityFailure(
            f"capability check {report.status}: " + "; ".join(report.warnings or ["unknown"])
        )
    if report.status == "failed":
        raise CapabilityFailure("model unreachable: " + "; ".join(report.warnings))
