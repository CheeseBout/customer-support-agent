"""The evaluation harness itself: it must not mark a right answer wrong, nor drift with the date.

These cover what made two shops' first reports misleading: a staff-only question that is rightly
refused counted as a miss, "15-day" not matching "15 days", an answer that says what the customer
can do later read as a contradiction, deadlines that moved every day, and one run taken for the
truth about a model that answers differently each time.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from support_agent.core.clock import NOW_ENV, frozen_clock, parse_instant
from support_agent.core.settings import AppConfig, EvalThresholds, Settings
from support_agent.evals import metrics as m
from support_agent.evals.dataset import Sample, check_dataset
from support_agent.evals.judge import JUDGE_PROMPT
from support_agent.evals.report import (
    EvalMeta,
    EvalReport,
    render_repeat_markdown,
    summarise_repeats,
)
from support_agent.evals.runner import (
    PipelineReport,
    RoutingReport,
    SampleRecord,
    score_sample,
    summarise_routing,
)
from support_agent.llm.usage import Usage
from support_agent.mcp_db.service import BusinessService
from support_agent.rag.pipeline import RagAnswer
from tests.conftest import ALICE
from tests.fakes import AgentFakeLLM
from tests.test_cli import _use_agent_llm, env, invoke  # noqa: F401  (fixtures and helpers)


def sample(sample_id: str = "s1", **expected: Any) -> Sample:
    return Sample.model_validate(
        {
            "id": sample_id,
            "lang": "en",
            "type": "trap",
            "principal": {"user_id": "u"},
            "input": "q",
            "expected": expected,
        }
    )


def answer(text: str = "ok", route: str = "policy", outcome: str = "answered") -> RagAnswer:
    return RagAnswer(answer=text, language="en", route=route, outcome=outcome)  # type: ignore[arg-type]


# --- several right results -------------------------------------------------------------------------------


def test_a_staff_only_question_may_be_refused_in_either_way():
    s = sample(route_any=["policy", "out_of_scope"], outcome_any=["no_info", "refused"])
    for route, outcome in (("out_of_scope", "refused"), ("policy", "no_info")):
        scores = score_sample(s, answer(route=route, outcome=outcome), None)
        assert scores.route_ok and scores.outcome_ok
    wrong = score_sample(s, answer(route="personal", outcome="answered"), None)
    assert wrong.route_ok is False and wrong.outcome_ok is False


def test_a_single_expected_result_works_as_before():
    s = sample(route="policy", outcome="answered")
    assert score_sample(s, answer(), None).route_ok is True
    assert score_sample(s, answer(route="personal"), None).route_ok is False
    nothing = score_sample(sample(route="policy"), answer(), None)
    assert nothing.outcome_ok is None  # nothing was asked about the outcome


def test_a_sample_that_lists_acceptable_results_expects_something():
    only_lists = sample(route_any=["policy", "out_of_scope"], outcome_any=["refused"])
    assert not any("expects neither" in p for p in check_dataset([only_lists]))
    empty = sample()
    assert any("expects neither" in p for p in check_dataset([empty]))


def test_routing_accuracy_counts_any_acceptable_route_as_right():
    ok = sample("a", route_any=["policy", "out_of_scope"])
    bad = sample("b", route_any=["policy", "out_of_scope"])
    report = summarise_routing("pipeline", [(ok, "out_of_scope"), (bad, "personal")])
    assert report.accuracy == 0.5 and report.failures == ["b"]
    assert report.confusion["out_of_scope"] == {"out_of_scope": 1}  # a right answer is no confusion
    assert isinstance(report, RoutingReport)


# --- wording that means the same ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("The 15-day exchange window ended on August 29.", "15 days"),
        ("You have a 15 day window.", "15 days"),
        ("Within 30-days of delivery", "30 days"),
        ("a 1-month warranty", "1 months"),
        ("Returns take 5 to 7 business days", "5 to 7 business days"),
    ],
)
def test_a_duration_matches_however_it_is_written(said: str, expected: str):
    assert m.fact_present(said, expected)


def test_a_different_number_of_days_still_does_not_match():
    assert not m.fact_present("The 7-day window ended.", "15 days")
    assert m.fact_present("Đổi trong 15 ngày", "15 ngày")  # Vietnamese is untouched


@pytest.mark.parametrize(
    "text",
    [
        "Đơn hàng chưa được giao thành công, nên bạn chưa thể trả hàng lúc này.",
        "Bạn chưa đủ điều kiện trả hàng. Khi nhận hàng bạn có thể yêu cầu trả.",
        "The order has not been delivered yet, so it cannot be returned. After delivery you can "
        "ask for a return.",
        "Your order hasn't been delivered, so you can't return it yet.",
    ],
)
def test_an_order_not_yet_delivered_reads_as_a_no(text: str):
    assert m.eligibility_polarity(text) == "no"


def test_a_plain_yes_is_still_a_yes():
    assert m.eligibility_polarity("Yes, you can return it within 3 more days.") == "yes"
    assert m.eligibility_polarity("Bạn còn đủ điều kiện đổi trả, hạn đến 13/10.") == "yes"


def test_the_judge_is_told_a_later_option_is_not_a_contradiction():
    assert "LATER" in JUDGE_PROMPT and "NOW" in JUDGE_PROMPT


# --- a frozen clock ------------------------------------------------------------------------------------------


def test_a_moment_without_a_zone_is_read_in_the_business_zone():
    assert parse_instant("2026-10-09T12:00:00", "Asia/Ho_Chi_Minh") == datetime(
        2026, 10, 9, 5, 0, tzinfo=UTC
    )
    assert parse_instant("2026-10-09T12:00:00+07:00") == datetime(2026, 10, 9, 5, 0, tzinfo=UTC)
    assert parse_instant("2026-10-09T05:00:00Z") == datetime(2026, 10, 9, 5, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        parse_instant("yesterday")


def test_a_frozen_clock_always_says_the_same_moment():
    clock = frozen_clock("2026-10-09T12:00:00+07:00")
    assert clock() == clock() == datetime(2026, 10, 9, 5, 0, tzinfo=UTC)


def test_the_frozen_moment_comes_from_the_environment_or_the_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(NOW_ENV, "2026-10-09T12:00:00+07:00")
    assert Settings(_env_file=None).support_agent_now == "2026-10-09T12:00:00+07:00"
    monkeypatch.delenv(NOW_ENV)
    assert Settings(_env_file=None).support_agent_now is None
    cfg = AppConfig.model_validate({"evals": {"as_of": "2026-10-09T12:00:00+07:00"}})
    assert cfg.evals.as_of is not None and cfg.evals.as_of.utcoffset() == timedelta(hours=7)
    assert AppConfig().evals.as_of is None  # real time unless a shop asks otherwise


async def test_a_frozen_clock_makes_a_deadline_stay_where_it_is(
    adapter: Any, app_config: AppConfig
):
    """The demo order 1234 was delivered four days ago and has a 7-day return window."""
    real = BusinessService(adapter, app_config.business_rules)
    assert (await real.check_return_eligibility(ALICE, "1234")).data["eligible"] is True
    later = datetime.now(UTC) + timedelta(days=30)
    frozen = BusinessService(adapter, app_config.business_rules, clock=lambda: later)
    result = await frozen.check_return_eligibility(ALICE, "1234")
    assert result.data["eligible"] is False and "WINDOW_EXPIRED" in result.data["reasons"]


# --- repeated runs -----------------------------------------------------------------------------------------------


def fake_report(name: str, metrics: dict[str, float | None], failed: list[str]) -> EvalReport:
    pipeline = PipelineReport.model_construct(
        n=4,
        errors=[],
        metrics=metrics,
        failures=[{"id": i, "problem": "x"} for i in failed],
        records=[SampleRecord(id=i) for i in ("a", "b", "c", "d")],
        usage=Usage(),
    )
    meta = EvalMeta(
        created_at="now",
        mode="full",
        dataset="d",
        n_samples=4,
        embedding_model="e",
        score_threshold=0.8,
        router="llm",
        run_name=name,
    )
    return EvalReport(meta=meta, pipeline=pipeline)


def test_repeated_runs_show_the_spread_and_which_questions_flip():
    runs = [
        fake_report("r1", {"answer": 0.75, "safety": 1.0, "business": 0.5}, ["a", "b"]),
        fake_report("r2", {"answer": 0.5, "safety": 1.0, "business": None}, ["a", "c"]),
        fake_report("r3", {"answer": 1.0, "safety": 1.0, "business": 0.5}, ["a"]),
    ]
    summary = summarise_repeats("shop", runs, EvalThresholds(answer=0.7, safety=1.0))
    answer_spread = summary.metrics["answer"]
    assert answer_spread.mean == pytest.approx(0.75)
    assert (answer_spread.low, answer_spread.high) == (0.5, 1.0)
    assert summary.metrics["business"].mean == 0.5  # a run that did not measure it is left out
    assert summary.always_failing == ["a"]  # failed every time
    assert summary.unstable == {"b": "2/3 runs passed", "c": "2/3 runs passed"}
    assert summary.repeats == 3 and summary.reports == ["r1", "r2", "r3"]
    by_name = {c.name: c for c in summary.thresholds}
    assert by_name["answer"].passed is True and by_name["safety"].passed is True


def test_the_summary_judges_the_mean_not_the_worst_run():
    runs = [fake_report("r1", {"safety": 1.0}, []), fake_report("r2", {"safety": 0.5}, [])]
    summary = summarise_repeats("shop", runs, EvalThresholds(safety=1.0))
    assert not summary.passed  # a mean of 0.75 is below the required 1.0
    text = render_repeat_markdown(summary)
    assert "2 runs" in text and "Mean" in text and "**FAIL**" in text
    assert "Passed in some runs only" in text


# --- the command ------------------------------------------------------------------------------------------------------


def test_repeating_a_run_writes_each_run_and_a_summary(env: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    invoke("ingest")
    reports = env / "evals" / "reports"
    reports.mkdir(parents=True)
    _use_agent_llm(monkeypatch, AgentFakeLLM([], route="chitchat"))
    result = invoke("eval", "--limit", "4", "--repeat", "2", "--name", "twice")
    assert (reports / "twice-r1.json").exists() and (reports / "twice-r2.json").exists()
    summary = json.loads((reports / "twice-summary.json").read_text(encoding="utf-8"))
    assert summary["repeats"] == 2 and summary["reports"] == ["twice-r1", "twice-r2"]
    assert "Evaluation summary: twice (2 runs)" in result.output
    assert not (reports / "twice.json").exists()  # the runs have their own names


def test_a_frozen_clock_is_recorded_in_the_report(env: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: F811
    invoke("ingest")
    reports = env / "evals" / "reports"
    reports.mkdir(parents=True)
    monkeypatch.setenv(NOW_ENV, "")  # restored afterwards, whatever the command sets
    _use_agent_llm(monkeypatch, AgentFakeLLM([], route="chitchat"))
    invoke("eval", "--limit", "4", "--as-of", "2026-10-09T12:00:00+07:00", "--name", "frozen")
    saved = json.loads((reports / "frozen.json").read_text(encoding="utf-8"))
    assert saved["meta"]["as_of"] == "2026-10-09T12:00:00+07:00"
    assert "frozen at 2026-10-09T12:00:00+07:00" in (reports / "frozen.md").read_text("utf-8")


def test_a_moment_that_is_not_a_date_is_refused(env: Path):  # noqa: F811
    from typer.testing import CliRunner

    from support_agent.cli import app

    result = CliRunner().invoke(app, ["eval", "--limit", "1", "--as-of", "yesterday"])
    assert result.exit_code != 0
