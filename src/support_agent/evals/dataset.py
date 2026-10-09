"""Evaluation dataset: JSONL, one sample per line (SPEC 15.2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from support_agent.core.principal import Principal
from support_agent.rag.pipeline import Outcome
from support_agent.rag.router import Route

SampleType = Literal["policy", "personal", "combined", "trap", "aftersales"]
DraftKind = Literal["refund", "return", "warranty", "order", "handoff"]

# An answer fact is a string, or a list of alternatives of which any one satisfies it.
Fact = str | list[str]


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Expected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Route | None = None
    outcome: Outcome | None = None
    tools: list[str] = Field(default_factory=list)  # tools that must have been called
    # At least one of these must have been called: for where a sensible agent has several ways in
    # (asking what is missing can start from a stock check or a search).
    tools_any: list[str] = Field(default_factory=list)
    answer_facts: list[Fact] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)  # acceptable doc_ids (retrieval)
    sections: list[str] = Field(default_factory=list)  # acceptable section substrings
    eligible: bool | None = None  # the rules engine's true verdict, for combined return questions
    no_policy_match: bool = False  # nothing in the knowledge base should match
    # Requests (aftersales samples). A sample that names no `draft_type` must end with nothing
    # proposed and nothing created.
    draft_type: DraftKind | None = None  # the confirmation the customer should be shown
    draft_amount: int | None = None  # its refundable amount, or the order total, in VND
    priority_review: bool | None = None
    summary_contains: list[str] = Field(default_factory=list)  # text the confirmation must show
    # What the simulated customer answers when asked to confirm; no decision = stop there.
    decision: Literal["approve", "reject", "edit"] | None = None
    edits: dict[str, str] = Field(default_factory=dict)  # with decision "edit"


class Sample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    lang: Literal["vi", "en"]
    type: SampleType
    tags: list[str] = Field(default_factory=list)
    subset: list[str] = Field(default_factory=list)
    principal: Principal
    history: list[HistoryTurn] = Field(default_factory=list)
    input: str
    expected: Expected

    @property
    def mutates(self) -> bool:
        """Creates a draft, so it must not run beside samples that propose the same request."""
        return self.expected.decision == "approve" or "after_approvals" in self.tags

    @property
    def safety_relevant(self) -> bool:
        return bool({"safety", "foreign_order", "injection", "indirect_injection"} & set(self.tags))


class DatasetError(ValueError):
    pass


def load_dataset(
    path: Path, *, subset: str | None = None, limit: int | None = None
) -> list[Sample]:
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    samples: list[Sample] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(Sample.model_validate(json.loads(line)))
        except ValueError as exc:
            raise DatasetError(f"{path}:{n}: {exc}") from exc
    if subset:
        samples = [s for s in samples if subset in s.subset]
    return samples[:limit] if limit else samples


def check_dataset(samples: list[Sample]) -> list[str]:
    """Structural problems that would make results meaningless (empty list = fine)."""
    problems: list[str] = []
    ids = [s.id for s in samples]
    problems += [f"duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    langs = {lang: sum(1 for s in samples if s.lang == lang) for lang in ("vi", "en")}
    total = max(len(samples), 1)
    if not 0.4 <= langs["vi"] / total <= 0.6:
        problems.append(f"VI/EN split is unbalanced: {langs}")
    if not all(s.type == "aftersales" for s in samples):  # the baseline needs all three kinds
        for kind in ("policy", "personal", "combined"):
            if not any(s.type == kind for s in samples):
                problems.append(f"no samples of type {kind!r}")
    # Approved drafts stay in the store, so two of them must not ask for the same thing.
    approved = [
        (s.principal.user_id, s.expected.draft_type, s.id)
        for s in samples
        if s.expected.decision == "approve"
    ]
    problems += [
        f"{b[2]}: approves what {a[2]} already approves"
        for i, a in enumerate(approved)
        for b in approved[i + 1 :]
        if a[:2] == b[:2] and a[1] in ("refund", "return")
    ]
    for s in samples:
        if s.expected.route is None and s.expected.outcome is None:
            problems.append(f"{s.id}: expects neither a route nor an outcome")
        answerable_policy = s.type == "policy" and s.expected.outcome == "answered"
        if answerable_policy and not s.expected.docs:
            problems.append(f"{s.id}: policy answer without expected docs")
    return problems
