"""Streaming events (SPEC 14.5). Nodes emit plain dicts; the API layer sees `AgentEvent`."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventKind = Literal[
    "session",  # first event: {session_id, message_id}
    "route",  # {route, language}
    "tool_start",  # {call_id, tool, args_summary}   (PII already masked)
    "tool_end",  # {call_id, tool, ok, duration_ms}  (never the raw tool data)
    "token",  # {text}: a piece of the answer, citation markers already removed
    "citation",  # {source, section}
    "replace",  # {text}: the output guardrail withdrew the text streamed so far; show this instead
    "interrupt",  # a human decision is needed (Phase 5)
    "done",  # {answer, outcome, usage, trace_id, ...}: always last on success
    "error",  # {code, message}: always last on failure
]


class AgentEvent(BaseModel):
    kind: EventKind
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.kind in ("done", "error", "interrupt")
