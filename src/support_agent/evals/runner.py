"""Run the three evaluations: retrieval (no LLM), routing, and the full pipeline."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from support_agent.core.principal import Principal, hash_user_id
from support_agent.core.settings import EvalThresholds
from support_agent.evals import metrics as m
from support_agent.evals.dataset import Sample
from support_agent.evals.judge import JudgeVerdict, LLMJudge
from support_agent.llm.usage import Usage
from support_agent.observability.langfuse import Tracing
from support_agent.rag.entities import extract_entities
from support_agent.rag.pipeline import RagAnswer
from support_agent.rag.retriever import Retriever
from support_agent.rag.router import heuristic_route

log = logging.getLogger(__name__)

# --- retrieval -----------------------------------------------------------------------


class GateReport(BaseModel):
    configured_threshold: float
    recall: float | None  # answerable questions that pass the gate at the configured threshold
    specificity: float | None  # unanswerable questions rejected at the configured threshold
    recommended_threshold: float | None
    n_positive: int
    n_negative: int
    lowest_positive: float | None
    highest_negative: float | None
    lowest_positives: list[tuple[str, float]] = Field(default_factory=list)
    highest_negatives: list[tuple[str, float]] = Field(default_factory=list)
    curve: list[dict[str, float]] = Field(default_factory=list)


class RetrievalReport(BaseModel):
    n_evaluated: int
    hit_at: dict[int, float | None]
    section_hit_at: dict[int, float | None]
    # Only documents in the other language than the question (FR-006: VI finds EN and back).
    cross_lingual_hit_at: dict[int, float | None]
    mrr: float | None
    by_lang: dict[str, dict[str, float | None]]
    gate: GateReport
    misses: list[str] = Field(default_factory=list)  # ids with no expected doc in the top-k


def evaluate_retrieval(
    retriever: Retriever,
    samples: list[Sample],
    *,
    ks: list[int],
    configured_threshold: float,
) -> RetrievalReport:
    """Ranking quality ignores the gate (threshold 0); the gate is analysed separately."""
    k_max = max(ks)
    doc_ranks: dict[str, int | None] = {}
    section_ranks: dict[str, int | None] = {}
    cross_ranks: dict[str, int | None] = {}
    positives: dict[str, float] = {}
    negatives: dict[str, float] = {}

    for s in samples:
        wants_docs = bool(s.expected.docs)
        # Unanswerable = nothing in the KB should match: uncovered policy questions, plus
        # off-topic and small talk (a second line of defence behind the router).
        unanswerable = s.expected.no_policy_match or s.expected.route in (
            "out_of_scope",
            "chitchat",
        )
        if not (wants_docs or unanswerable):
            continue
        hits = retriever.retrieve_sync(s.input, top_k=k_max, threshold=0.0)
        best = max((h.score for h in hits), default=0.0)  # what the query-level gate sees
        if wants_docs:
            positives[s.id] = best
            doc_ranks[s.id], section_ranks[s.id] = m.first_rank(
                [(h.doc_id, h.section) for h in hits], s.expected.docs, s.expected.sections
            )
            wide = retriever.retrieve_sync(s.input, top_k=40, threshold=0.0)
            cross_ranks[s.id], _ = m.first_rank(
                [(h.doc_id, h.section) for h in wide if h.lang != s.lang], s.expected.docs, []
            )
        else:
            negatives[s.id] = best

    ids = list(doc_ranks)
    by_id = {s.id: s for s in samples}

    def rate(ranks: dict[str, int | None], k: int, only: list[str]) -> float | None:
        return m.mean([m.hit_at_k(ranks[i], k) for i in only])

    with_sections = [i for i in ids if by_id[i].expected.sections]
    by_lang: dict[str, dict[str, float | None]] = {}
    for lang in ("en", "vi"):
        lang_ids = [i for i in ids if by_id[i].lang == lang]
        by_lang[lang] = {
            "hit@1": rate(doc_ranks, 1, lang_ids),
            f"hit@{k_max}": rate(doc_ranks, k_max, lang_ids),
            "mrr": m.mean([m.reciprocal_rank(doc_ranks[i]) for i in lang_ids]),
        }

    pos_scores, neg_scores = list(positives.values()), list(negatives.values())
    thresholds = [round(0.60 + i * 0.01, 2) for i in range(0, 36)]
    curve = m.gate_curve(pos_scores, neg_scores, thresholds)
    at_config = m.gate_curve(pos_scores, neg_scores, [configured_threshold])[0]

    return RetrievalReport(
        n_evaluated=len(ids),
        hit_at={k: rate(doc_ranks, k, ids) for k in ks},
        section_hit_at={k: rate(section_ranks, k, with_sections) for k in ks},
        cross_lingual_hit_at={k: rate(cross_ranks, k, ids) for k in (1, 3)},
        mrr=m.mean([m.reciprocal_rank(r) for r in doc_ranks.values()]),
        by_lang=by_lang,
        gate=GateReport(
            configured_threshold=configured_threshold,
            recall=at_config.recall if pos_scores else None,
            specificity=at_config.specificity if neg_scores else None,
            recommended_threshold=m.recommend_threshold(pos_scores, neg_scores, curve),
            n_positive=len(pos_scores),
            n_negative=len(neg_scores),
            lowest_positive=min(pos_scores) if pos_scores else None,
            highest_negative=max(neg_scores) if neg_scores else None,
            lowest_positives=sorted(positives.items(), key=lambda kv: kv[1])[:5],
            highest_negatives=sorted(negatives.items(), key=lambda kv: -kv[1])[:5],
            curve=[
                {"threshold": p.threshold, "recall": p.recall, "specificity": p.specificity}
                for p in curve
            ],
        ),
        misses=[i for i in ids if not m.hit_at_k(doc_ranks[i], k_max)],
    )


# --- routing (heuristic fallback, no LLM) -------------------------------------------


class RoutingReport(BaseModel):
    source: str  # "heuristic" | "llm"
    n: int
    accuracy: float | None
    by_type: dict[str, float | None]
    confusion: dict[str, dict[str, int]]
    failures: list[str] = Field(default_factory=list)


def evaluate_heuristic_routing(samples: list[Sample]) -> RoutingReport:
    """How good the no-LLM fallback router is: the floor the LLM router must beat."""
    rows = [
        (s, heuristic_route(s.input, extract_entities(s.input)).route)
        for s in samples
        if s.expected.route
    ]
    return summarise_routing("heuristic", rows)


def summarise_routing(source: str, rows: Sequence[tuple[Sample, str | None]]) -> RoutingReport:
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_type: dict[str, list[bool]] = defaultdict(list)
    for s, actual in rows:
        ok = actual == s.expected.route
        confusion[str(s.expected.route)][str(actual)] += 1
        by_type[s.type].append(ok)
    return RoutingReport(
        source=source,
        n=len(rows),
        accuracy=m.mean([actual == s.expected.route for s, actual in rows]),
        by_type={k: m.mean(v) for k, v in sorted(by_type.items())},
        confusion={k: dict(v) for k, v in confusion.items()},
        failures=[s.id for s, actual in rows if actual != s.expected.route],
    )


# --- full pipeline ------------------------------------------------------------------


@dataclass
class SampleScores:
    route_ok: bool | None = None
    outcome_ok: bool | None = None
    tools_ok: bool | None = None
    facts_found: int = 0
    facts_total: int = 0
    forbidden: list[str] = field(default_factory=list)
    language_ok: bool | None = None
    polarity_ok: bool | None = None
    judge_ok: bool | None = None  # None = not judged
    draft_ok: bool | None = None  # requests: right confirmation, right drafts (None = n/a)
    answer_ok: bool = False
    safety_ok: bool | None = None

    @property
    def facts_ok(self) -> bool:
        return self.facts_found == self.facts_total

    @property
    def business_ok(self) -> bool | None:
        """Business correctness: the eligibility verdict and the request that was proposed."""
        checks = [c for c in (self.polarity_ok, self.draft_ok) if c is not None]
        return all(checks) if checks else None


@dataclass
class SampleResult:
    sample: Sample
    answer: RagAnswer | None
    scores: SampleScores
    verdict: JudgeVerdict | None = None
    error: str | None = None


def score_sample(sample: Sample, answer: RagAnswer, verdict: JudgeVerdict | None) -> SampleScores:
    exp = sample.expected
    sc = SampleScores()
    sc.route_ok = None if exp.route is None else answer.route == exp.route
    sc.outcome_ok = None if exp.outcome is None else answer.outcome == exp.outcome
    # Right data tools called (and none for a policy question), and policy text actually
    # retrieved whenever the answer needs it.
    searched = not exp.docs or bool(answer.retrieved)
    called = m.data_tools(answer.tool_calls)
    sc.tools_ok = (
        (m.tools_ok(exp.tools, called) if exp.tools or not exp.tools_any else True)
        and searched
        and (not exp.tools_any or any(t in called for t in exp.tools_any))
    )
    sc.facts_found, sc.facts_total = m.facts_satisfied(answer.answer, exp.answer_facts)
    sc.forbidden = m.forbidden_hits(answer.answer, exp.must_not)
    # A confirmation is a form, not a reply: it has no language to judge.
    sc.language_ok = (
        None if answer.outcome == "confirmation" else m.language_ok(answer.answer, sample.lang)
    )
    if exp.eligible is not None:
        sc.polarity_ok = m.eligibility_polarity(answer.answer) == ("yes" if exp.eligible else "no")
    if sample.type == "aftersales" or exp.draft_type is not None:
        sc.draft_ok = m.draft_ok(
            expected_type=exp.draft_type,
            amount=exp.draft_amount,
            priority=exp.priority_review,
            contains=exp.summary_contains,
            decision=exp.decision,
            interrupt=answer.interrupt,
            created=answer.drafts,
        )
    if verdict is not None and verdict.correct is not None:
        sc.judge_ok = verdict.correct
    sc.answer_ok = (
        sc.outcome_ok is not False
        and sc.draft_ok is not False
        and sc.facts_ok
        and not sc.forbidden
        and sc.polarity_ok is not False
        and sc.judge_ok is not False
    )
    if sample.safety_relevant:
        sc.safety_ok = not sc.forbidden and sc.outcome_ok is not False and sc.draft_ok is not False
    return sc


class Answerer(Protocol):
    """What the evaluation needs from a system under test: the Phase A pipeline and the agent
    both fit, which is what lets one dataset compare them."""

    retriever: Retriever

    async def answer(
        self,
        question: str,
        principal: Principal,
        *,
        language: str | None = ...,
        history: list[BaseMessage] | None = ...,
        callbacks: Any = ...,
        metadata: dict[str, Any] | None = ...,
        trace_id: str | None = ...,
    ) -> RagAnswer: ...


class Decider(Protocol):
    """A system that can stop for a confirmation and carry on with the customer's answer."""

    async def decide(
        self,
        principal: Principal,
        shown: RagAnswer,
        decision: str,
        *,
        edits: dict[str, Any] | None = ...,
        callbacks: Any = ...,
        trace_id: str | None = ...,
    ) -> RagAnswer: ...


def _history(sample: Sample) -> list[BaseMessage]:
    return [
        HumanMessage(content=t.content) if t.role == "user" else AIMessage(content=t.content)
        for t in sample.history
    ]


async def run_pipeline_eval(
    rag: Answerer,
    samples: list[Sample],
    *,
    run_name: str,
    judge: LLMJudge | None = None,
    tracing: Tracing | None = None,
    concurrency: int = 1,
) -> list[SampleResult]:
    tracing = tracing or Tracing()
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(sample: Sample) -> SampleResult:
        async with gate:
            trace_id = tracing.trace_id(f"{run_name}:{sample.id}")
            handler = tracing.handler(trace_id)
            try:
                answer = await rag.answer(
                    sample.input,
                    sample.principal,
                    language="auto",
                    history=_history(sample) or None,
                    callbacks=[handler] if handler else None,
                    trace_id=trace_id,
                    metadata={
                        "langfuse_session_id": run_name,
                        "langfuse_user_id": hash_user_id(sample.principal.user_id),
                        "langfuse_tags": ["eval", sample.type, sample.lang],
                        "sample_id": sample.id,
                    },
                )
            except Exception as exc:  # the harness must survive any single sample
                log.exception("sample %s crashed", sample.id)
                return SampleResult(
                    sample, None, SampleScores(), error=f"{type(exc).__name__}: {exc}"
                )

            if answer.outcome == "confirmation" and sample.expected.decision:
                decide = getattr(rag, "decide", None)
                if decide is not None:
                    try:
                        answer = await decide(
                            sample.principal,
                            answer,
                            sample.expected.decision,
                            edits=sample.expected.edits or None,
                            callbacks=[handler] if handler else None,
                            trace_id=trace_id,
                        )
                    except Exception as exc:
                        log.exception("sample %s crashed while confirming", sample.id)
                        return SampleResult(
                            sample, None, SampleScores(), error=f"{type(exc).__name__}: {exc}"
                        )

            verdict = None
            judgeable = answer.outcome == "answered" and (
                sample.expected.answer_facts or sample.expected.eligible is not None
            )
            if judge and judgeable:
                verdict = await judge.grade(sample, answer.answer)
            scores = score_sample(sample, answer, verdict)
            tracing.score(trace_id, "answer_ok", scores.answer_ok)
            if scores.route_ok is not None:
                tracing.score(trace_id, "route_ok", scores.route_ok)
            if scores.safety_ok is not None:
                tracing.score(trace_id, "safety_ok", scores.safety_ok)
            return SampleResult(sample, answer, scores, verdict)

    # Samples that create drafts (or read them back) go last, one at a time, in file order:
    # what an earlier one stores must not change what a later one is asked to propose.
    parallel = [s for s in samples if not s.mutates]
    serial = [s for s in samples if s.mutates]
    done = dict(
        zip(
            (s.id for s in parallel), await asyncio.gather(*(one(s) for s in parallel)), strict=True
        )
    )
    for s in serial:
        done[s.id] = await one(s)
    tracing.flush()
    return [done[s.id] for s in samples]


class PipelineReport(BaseModel):
    n: int
    errors: list[str] = Field(default_factory=list)
    routing: RoutingReport
    metrics: dict[str, float | None]  # routing, trajectory, answer, business, safety, language
    by_type: dict[str, dict[str, float | None]]
    by_lang: dict[str, dict[str, float | None]]
    judged: int
    usage: Usage
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    failures: list[dict[str, object]] = Field(default_factory=list)
    # One record per question, enough to re-score without calling any model (see rescore).
    records: list[SampleRecord] = Field(default_factory=list)


class SampleRecord(BaseModel):
    id: str
    answer: RagAnswer | None = None
    verdict: JudgeVerdict | None = None
    error: str | None = None


PipelineReport.model_rebuild()


def rescore_pipeline(saved: dict[str, Any], samples: list[Sample]) -> PipelineReport:
    """Score a saved run again against the current dataset and metrics. No model is called.

    Use it after correcting an expectation or a metric: the answers are what the system really
    produced, only the grading changes.
    """
    by_id = {s.id: s for s in samples}
    results: list[SampleResult] = []
    for raw in (saved.get("pipeline") or {}).get("records", []):
        record = SampleRecord.model_validate(raw)
        sample = by_id.get(record.id)
        if sample is None:
            continue
        if record.answer is None:
            results.append(SampleResult(sample, None, SampleScores(), error=record.error))
            continue
        scores = score_sample(sample, record.answer, record.verdict)
        results.append(SampleResult(sample, record.answer, scores, record.verdict))
    return summarise_pipeline(results)


def summarise_pipeline(results: list[SampleResult]) -> PipelineReport:
    done = [r for r in results if r.answer is not None]

    def collect(
        pick: Callable[[SampleResult], bool | None], rs: list[SampleResult] | None = None
    ) -> float | None:
        vals = [pick(r) for r in (rs if rs is not None else done)]
        return m.mean([v for v in vals if v is not None])

    def block(rs: list[SampleResult]) -> dict[str, float | None]:
        return {
            "routing": collect(lambda r: r.scores.route_ok, rs),
            "trajectory": collect(lambda r: r.scores.tools_ok, rs),
            "answer": collect(lambda r: r.scores.answer_ok, rs),
            "language": collect(lambda r: r.scores.language_ok, rs),
        }

    usage = Usage()
    for r in done:
        assert r.answer is not None
        usage.add(r.answer.usage)
    latencies = [float(r.answer.latency_ms) for r in done if r.answer]

    failures: list[dict[str, object]] = []
    for r in results:
        if r.answer is None:
            failures.append({"id": r.sample.id, "problem": f"crashed: {r.error}"})
            continue
        sc, exp = r.scores, r.sample.expected
        problems = []
        if sc.route_ok is False:
            problems.append(f"route {r.answer.route!r} != {exp.route!r}")
        if sc.outcome_ok is False:
            problems.append(f"outcome {r.answer.outcome!r} != {exp.outcome!r}")
        if sc.tools_ok is False:
            if exp.docs and not r.answer.retrieved:
                problems.append("policy documents were needed but none were retrieved")
            else:
                problems.append(
                    f"tools {m.data_tools(r.answer.tool_calls)} vs expected {exp.tools}"
                )
        if not sc.facts_ok:
            problems.append(f"facts {sc.facts_found}/{sc.facts_total}")
        if sc.forbidden:
            problems.append(f"forbidden text present: {sc.forbidden}")
        if sc.polarity_ok is False:
            problems.append(f"eligibility verdict not as expected ({exp.eligible})")
        if sc.draft_ok is False:
            shown = r.answer.interrupt or {}
            problems.append(
                f"request: expected {exp.draft_type or 'nothing'} (decision {exp.decision}), "
                f"got {shown.get('draft_type') or 'no confirmation'}"
                f"{' ' + str(shown.get('summary')) if shown else ''}, "
                f"{len(r.answer.drafts)} draft(s) created"
            )
        if sc.judge_ok is False and r.verdict:
            problems.append(f"judge: {r.verdict.reason}")
        if problems:
            failures.append(
                {
                    "id": r.sample.id,
                    "input": r.sample.input,
                    "answer": r.answer.answer[:300],
                    "problem": "; ".join(problems),
                }
            )

    types = sorted({r.sample.type for r in done})
    langs = sorted({r.sample.lang for r in done})
    return PipelineReport(
        n=len(results),
        errors=[f"{r.sample.id}: {r.error}" for r in results if r.error],
        routing=summarise_routing(
            "pipeline",
            [
                (r.sample, r.answer.route if r.answer else None)
                for r in results
                if r.sample.expected.route
            ],
        ),
        metrics={
            "routing": collect(lambda r: r.scores.route_ok),
            "trajectory": collect(lambda r: r.scores.tools_ok),
            "answer": collect(lambda r: r.scores.answer_ok),
            "business": collect(lambda r: r.scores.business_ok),
            "safety": collect(lambda r: r.scores.safety_ok),
            "language": collect(lambda r: r.scores.language_ok),
        },
        by_type={t: block([r for r in done if r.sample.type == t]) for t in types},
        by_lang={lang: block([r for r in done if r.sample.lang == lang]) for lang in langs},
        judged=sum(1 for r in done if r.scores.judge_ok is not None),
        usage=usage,
        latency_ms_p50=m.percentile(latencies, 0.5),
        latency_ms_p95=m.percentile(latencies, 0.95),
        failures=failures,
        records=[
            SampleRecord(id=r.sample.id, answer=r.answer, verdict=r.verdict, error=r.error)
            for r in results
        ],
    )


# --- thresholds ---------------------------------------------------------------------


class ThresholdCheck(BaseModel):
    name: str
    value: float | None
    threshold: float
    passed: bool | None  # None = metric not measured in this mode


def check_thresholds(
    values: dict[str, float | None], thresholds: EvalThresholds
) -> list[ThresholdCheck]:
    out = []
    for name in ("routing", "trajectory", "answer", "business", "safety"):
        value = values.get(name)
        limit = getattr(thresholds, name)
        out.append(
            ThresholdCheck(
                name=name,
                value=value,
                threshold=limit,
                passed=None if value is None else value >= limit,
            )
        )
    return out
