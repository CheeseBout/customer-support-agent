"""Evaluation report: structured (JSON) and human-readable (Markdown)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from support_agent.evals.runner import (
    PipelineReport,
    RetrievalReport,
    RoutingReport,
    ThresholdCheck,
)


class EvalMeta(BaseModel):
    created_at: str
    mode: str  # "offline" (no LLM) | "full"
    dataset: str
    n_samples: int
    subset: str | None = None
    provider: str | None = None
    model: str | None = None
    embedding_model: str
    score_threshold: float
    router: str  # "llm" | "heuristic"
    judge: bool = False
    run_name: str
    tracing: bool = False
    trace_session: str | None = None  # Langfuse session id for this run
    engine: str = "pipeline"  # "pipeline" (Phase A) | "agent" (Phase 4)
    prompt_version: str | None = None
    run_timeout_seconds: float | None = None  # per-question limit this run used
    rescored_from: str | None = None  # set when graded again from a saved run
    as_of: str | None = None  # the frozen "now" of the run, if any (see evals.as_of)


class MetricDelta(BaseModel):
    name: str
    baseline: float | None
    current: float | None
    regressed: bool  # current is below baseline by more than the tolerance


class Comparison(BaseModel):
    """This run against an earlier one (PLAN Phase 4: the agent must not do worse)."""

    against: str  # report name of the baseline
    baseline_engine: str
    tolerance: float
    metrics: list[MetricDelta]
    newly_failing: list[str]  # samples that passed in the baseline and fail now
    newly_passing: list[str]

    @property
    def regressed(self) -> bool:
        return any(m.regressed for m in self.metrics)


class EvalReport(BaseModel):
    meta: EvalMeta
    retrieval: RetrievalReport | None = None
    routing: RoutingReport | None = None
    pipeline: PipelineReport | None = None
    thresholds: list[ThresholdCheck] = []
    comparison: Comparison | None = None

    @property
    def passed(self) -> bool:
        no_regression = self.comparison is None or not self.comparison.regressed
        return no_regression and all(c.passed is not False for c in self.thresholds)


def compare_reports(
    current: PipelineReport, baseline: dict[str, Any], *, against: str, tolerance: float = 0.0
) -> Comparison:
    """Compare against a saved report (its JSON form). A metric regresses if it drops by more
    than `tolerance`; unmeasured metrics never count as regressions."""
    base_pipeline = baseline.get("pipeline") or {}
    base_metrics: dict[str, float | None] = base_pipeline.get("metrics", {})
    deltas = []
    for name, now in current.metrics.items():
        before = base_metrics.get(name)
        regressed = now is not None and before is not None and now < before - tolerance
        deltas.append(MetricDelta(name=name, baseline=before, current=now, regressed=regressed))
    before_failing = {f["id"] for f in base_pipeline.get("failures", [])}
    now_failing = {str(f["id"]) for f in current.failures}
    return Comparison(
        against=against,
        baseline_engine=baseline.get("meta", {}).get("engine", "pipeline"),
        tolerance=tolerance,
        metrics=deltas,
        newly_failing=sorted(now_failing - before_failing),
        newly_passing=sorted(before_failing - now_failing),
    )


def trace_session(run_name: str, now: datetime | None = None) -> str:
    """A tracing session unique to one run, so re-running never overwrites earlier traces."""
    return f"{run_name}-{(now or datetime.now(UTC)):%Y%m%d-%H%M%S}"


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out + [""]


def render_markdown(r: EvalReport) -> str:
    meta = r.meta
    lines = [f"# Evaluation report: {meta.run_name}", ""]
    lines += _table(
        ["Field", "Value"],
        [
            ["Generated", meta.created_at],
            ["Mode", meta.mode],
            [
                "Dataset",
                f"{meta.dataset} ({meta.n_samples} samples"
                + (f", subset `{meta.subset}`)" if meta.subset else ")"),
            ],
            ["LLM", f"{meta.provider} / {meta.model}" if meta.model else "not used (offline)"],
            ["Embeddings", meta.embedding_model],
            ["Retrieval score threshold", str(meta.score_threshold)],
            [
                "Engine",
                meta.engine + (f" (prompt {meta.prompt_version})" if meta.prompt_version else ""),
            ],
            [
                "Per-question time limit",
                f"{meta.run_timeout_seconds:g} s" if meta.run_timeout_seconds else "n/a",
            ],
            ["Rescored from", f"`{meta.rescored_from}` (no model was called)"]
            if meta.rescored_from
            else ["Rescored from", "no (graded live)"],
            ["Clock", f"frozen at {meta.as_of}" if meta.as_of else "real time"],
            ["Router", meta.router],
            ["LLM judge", "yes" if meta.judge else "no"],
            ["Langfuse tracing", f"yes (session `{meta.trace_session}`)" if meta.tracing else "no"],
        ],
    )

    if meta.mode == "offline":
        lines += [
            "> Offline run: no LLM was called. Retrieval and the heuristic router are measured; "
            "answer quality, trajectory, business correctness and safety need `support-agent eval` "
            "with a provider key.",
            "",
        ]

    if r.retrieval:
        rt = r.retrieval
        lines += [
            "## Retrieval",
            "",
            f"Evaluated on {rt.n_evaluated} answerable questions (gate disabled).",
            "",
        ]
        lines += _table(
            ["Metric", "Value"],
            [[f"hit@{k}", _pct(v)] for k, v in rt.hit_at.items()]
            + [["MRR", _num(rt.mrr)]]
            + [[f"section hit@{k}", _pct(v)] for k, v in rt.section_hit_at.items()]
            + [
                [f"cross-language hit@{k} (other-language docs only)", _pct(v)]
                for k, v in rt.cross_lingual_hit_at.items()
            ],
        )
        lines += _table(
            ["Question language", *next(iter(rt.by_lang.values())).keys()],
            [
                [lang, *[_num(v) if k == "mrr" else _pct(v) for k, v in vals.items()]]
                for lang, vals in rt.by_lang.items()
            ],
        )
        g = rt.gate
        lines += [
            "### Score-threshold calibration",
            "",
            f"{g.n_positive} answerable and {g.n_negative} unanswerable questions "
            f"(not covered by the policies, off-topic, small talk).",
            "",
        ]
        lines += _table(
            ["", "Value"],
            [
                ["Configured threshold", str(g.configured_threshold)],
                ["Answerable questions passing it (recall)", _pct(g.recall)],
                ["Unanswerable questions rejected (specificity)", _pct(g.specificity)],
                ["Lowest score of an answerable question", _num(g.lowest_positive)],
                ["Highest score of an unanswerable question", _num(g.highest_negative)],
                ["**Recommended threshold**", _num(g.recommended_threshold)],
            ],
        )
        if g.lowest_positives or g.highest_negatives:
            lines += _table(
                ["Closest to the gate", "Question id", "Best score"],
                [["answerable (lowest)", f"`{i}`", _num(v)] for i, v in g.lowest_positives]
                + [["unanswerable (highest)", f"`{i}`", _num(v)] for i, v in g.highest_negatives],
            )
        if g.curve:
            lines += _table(
                ["threshold", "recall", "specificity"],
                [
                    [f"{p['threshold']:.2f}", _pct(p["recall"]), _pct(p["specificity"])]
                    for p in g.curve
                    if round(p["threshold"] * 100) % 2 == 0 and 0.70 <= p["threshold"] <= 0.90
                ],
            )
        if rt.misses:
            lines += [
                "Questions whose expected document was not in the top results: "
                + ", ".join(f"`{i}`" for i in rt.misses),
                "",
            ]

    routing = r.pipeline.routing if r.pipeline else r.routing
    if routing:
        lines += [
            "## Routing",
            "",
            f"Source: **{routing.source}**. Accuracy {_pct(routing.accuracy)} "
            f"over {routing.n} samples.",
            "",
        ]
        lines += _table(["Type", "Accuracy"], [[t, _pct(v)] for t, v in routing.by_type.items()])
        labels = sorted(
            {*routing.confusion, *(a for row in routing.confusion.values() for a in row)}
        )
        lines += _table(
            ["expected \\ actual", *labels],
            [
                [exp, *[str(routing.confusion.get(exp, {}).get(a, 0)) for a in labels]]
                for exp in labels
                if exp in routing.confusion
            ],
        )
        if routing.failures:
            lines += ["Mis-routed: " + ", ".join(f"`{i}`" for i in routing.failures), ""]

    if r.pipeline:
        p = r.pipeline
        lines += ["## Pipeline", ""]
        lines += _table(["Metric", "Value"], [[k, _pct(v)] for k, v in p.metrics.items()])
        lines += _table(
            ["Type", "routing", "trajectory", "answer", "language"],
            [
                [t, *[_pct(b[k]) for k in ("routing", "trajectory", "answer", "language")]]
                for t, b in p.by_type.items()
            ],
        )
        lines += _table(
            ["Language", "routing", "trajectory", "answer", "language"],
            [
                [t, *[_pct(b[k]) for k in ("routing", "trajectory", "answer", "language")]]
                for t, b in p.by_lang.items()
            ],
        )
        lines += [
            f"Tokens: {p.usage.input_tokens} in / {p.usage.output_tokens} out. "
            f"Latency p50 {_num(p.latency_ms_p50, 0)} ms, p95 {_num(p.latency_ms_p95, 0)} ms. "
            f"Judged by LLM: {p.judged}.",
            "",
        ]
        if p.errors:
            lines += ["### Crashed samples", ""] + [f"- {e}" for e in p.errors] + [""]
        if p.failures:
            lines += [f"### Failures ({len(p.failures)})", ""]
            for f in p.failures[:25]:
                lines.append(f"- `{f['id']}`: {f['problem']}")
            if len(p.failures) > 25:
                lines.append(f"- ... and {len(p.failures) - 25} more (see the JSON report)")
            lines.append("")

    if r.comparison:
        c = r.comparison
        verdict = "**REGRESSION**" if c.regressed else "no regression"
        lines += [
            f"## Comparison with `{c.against}` ({c.baseline_engine} engine): {verdict}",
            "",
        ]
        lines += _table(
            ["Metric", "Baseline", "This run", "Change"],
            [
                [
                    m.name,
                    _pct(m.baseline),
                    _pct(m.current),
                    "n/a"
                    if m.baseline is None or m.current is None
                    else f"{(m.current - m.baseline) * 100:+.1f} pts"
                    + (" **worse**" if m.regressed else ""),
                ]
                for m in c.metrics
            ],
        )
        if c.newly_failing:
            lines += [
                "Now failing, passed before: " + ", ".join(f"`{i}`" for i in c.newly_failing),
                "",
            ]
        if c.newly_passing:
            lines += [
                "Now passing, failed before: " + ", ".join(f"`{i}`" for i in c.newly_passing),
                "",
            ]

    lines += ["## Thresholds", ""]
    lines += _table(
        ["Metric", "Value", "Required", "Result"],
        [
            [
                c.name,
                _pct(c.value),
                f">= {_pct(c.threshold)}",
                "not measured" if c.passed is None else ("PASS" if c.passed else "**FAIL**"),
            ]
            for c in r.thresholds
        ],
    )
    return "\n".join(lines).rstrip() + "\n"


class Spread(BaseModel):
    """One metric over several runs of the same dataset."""

    mean: float | None
    low: float | None
    high: float | None
    values: list[float | None]


class RepeatSummary(BaseModel):
    """Several runs of the same dataset. A model answers a little differently every time, so one
    run cannot tell a real change from luck: the spread can."""

    run_name: str
    repeats: int
    n_samples: int
    reports: list[str]
    metrics: dict[str, Spread]
    always_failing: list[str]
    unstable: dict[str, str]  # sample -> "2/3 runs passed": noise until shown otherwise
    thresholds: list[ThresholdCheck] = []

    @property
    def passed(self) -> bool:
        return all(c.passed is not False for c in self.thresholds)


def summarise_repeats(
    run_name: str, reports: list[EvalReport], thresholds: Any | None = None
) -> RepeatSummary:
    from support_agent.evals.runner import check_thresholds

    pipelines = [r.pipeline for r in reports if r.pipeline is not None]
    names = list(pipelines[0].metrics) if pipelines else []
    metrics: dict[str, Spread] = {}
    for name in names:
        values = [p.metrics.get(name) for p in pipelines]
        known = [v for v in values if v is not None]
        metrics[name] = Spread(
            mean=sum(known) / len(known) if known else None,
            low=min(known) if known else None,
            high=max(known) if known else None,
            values=values,
        )
    failed_in = [{str(f["id"]) for f in p.failures} for p in pipelines]
    ids = [rec.id for rec in pipelines[0].records] if pipelines else []
    always, unstable = [], {}
    for sample_id in ids:
        passes = sum(1 for failed in failed_in if sample_id not in failed)
        if passes == 0:
            always.append(sample_id)
        elif passes < len(pipelines):
            unstable[sample_id] = f"{passes}/{len(pipelines)} runs passed"
    mean_metrics: dict[str, float | None] = {k: v.mean for k, v in metrics.items()}
    return RepeatSummary(
        run_name=run_name,
        repeats=len(pipelines),
        n_samples=reports[0].meta.n_samples if reports else 0,
        reports=[r.meta.run_name for r in reports],
        metrics=metrics,
        always_failing=always,
        unstable=unstable,
        thresholds=check_thresholds(mean_metrics, thresholds) if thresholds is not None else [],
    )


def render_repeat_markdown(s: RepeatSummary) -> str:
    lines = [f"# Evaluation summary: {s.run_name} ({s.repeats} runs)", ""]
    lines += [
        f"{s.n_samples} questions, run {s.repeats} times. Runs: "
        + ", ".join(f"`{n}`" for n in s.reports)
        + ".",
        "",
        "## Metrics (mean, and the lowest and highest run)",
        "",
    ]
    rows = []
    for name, sp in s.metrics.items():
        runs = ", ".join(_pct(v) for v in sp.values)
        rows.append([name, _pct(sp.mean), _pct(sp.low), _pct(sp.high), runs])
    lines += _table(["Metric", "Mean", "Lowest", "Highest", "Each run"], rows)
    if s.thresholds:
        lines += ["", "## Thresholds (on the mean)", ""]
        lines += _table(
            ["Metric", "Mean", "Required", "Result"],
            [
                [
                    c.name,
                    _pct(c.value),
                    f">= {c.threshold * 100:.1f}%",
                    "n/a" if c.passed is None else ("PASS" if c.passed else "**FAIL**"),
                ]
                for c in s.thresholds
            ],
        )
    lines += ["", "## Questions"]
    lines += ["", "**Failed in every run** (a real problem, or a wrong expectation):", ""]
    lines += [f"- `{i}`" for i in s.always_failing] or ["- none"]
    lines += ["", "**Passed in some runs only** (noise: do not read one run as a change):", ""]
    lines += [f"- `{i}`: {why}" for i, why in s.unstable.items()] or ["- none"]
    return "\n".join(lines) + "\n"


def write_repeat_summary(s: RepeatSummary, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    md, js = directory / f"{name}-summary.md", directory / f"{name}-summary.json"
    md.write_text(render_repeat_markdown(s), encoding="utf-8")
    js.write_text(s.model_dump_json(indent=2), encoding="utf-8")
    return md, js


def write_report(report: EvalReport, directory: Path, name: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    md, js = directory / f"{name}.md", directory / f"{name}.json"
    md.write_text(render_markdown(report), encoding="utf-8")
    js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return md, js
