from __future__ import annotations

import json
from pathlib import Path

import pytest

from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig, EvalThresholds, RetrievalConfig
from support_agent.evals import metrics as m
from support_agent.evals.dataset import (
    DatasetError,
    Expected,
    Sample,
    check_dataset,
    load_dataset,
)
from support_agent.evals.judge import LLMJudge
from support_agent.evals.report import EvalMeta, EvalReport, render_markdown, write_report
from support_agent.evals.runner import (
    check_thresholds,
    evaluate_heuristic_routing,
    evaluate_retrieval,
    run_pipeline_eval,
    score_sample,
    summarise_pipeline,
)
from support_agent.observability.langfuse import Tracing, create_tracing
from support_agent.rag.index import VectorStore
from support_agent.rag.ingest import ingest
from support_agent.rag.pipeline import RagAnswer, SupportRAG
from support_agent.rag.retriever import Retriever
from support_agent.tools.client import DomainToolClient
from tests.conftest import ROOT
from tests.fakes import (
    FakeSparse,
    HashingEmbeddings,
    ScriptedChatModel,
    SupportFakeLLM,
    ai_json,
)

DATASET = ROOT / "evals" / "datasets" / "baseline.jsonl"


# --- the shipped dataset -------------------------------------------------------------------------


def test_shipped_dataset_is_valid_balanced_and_the_right_size():
    samples = load_dataset(DATASET)
    assert 60 <= len(samples) <= 80  # PLAN Phase 3
    assert check_dataset(samples) == []
    langs = [s.lang for s in samples]
    assert abs(langs.count("vi") - langs.count("en")) <= 4
    assert {s.type for s in samples} == {"policy", "personal", "combined", "trap"}


def test_dataset_contains_the_required_trap_cases():
    tags = {t for s in load_dataset(DATASET) for t in s.tags}
    assert {"not_found", "foreign_order", "out_of_scope", "injection", "indirect_injection"} <= tags


def test_ci_subset_is_a_small_slice_covering_every_type():
    ci = load_dataset(DATASET, subset="ci")
    assert 20 <= len(ci) <= 30
    assert {s.type for s in ci} == {"policy", "personal", "combined", "trap"}
    assert {s.lang for s in ci} == {"vi", "en"}


def test_foreign_order_samples_use_a_non_owner_and_forbid_leaks():
    foreign = [s for s in load_dataset(DATASET) if "foreign_order" in s.tags]
    assert len(foreign) >= 4
    assert all(s.expected.outcome == "not_found" and s.expected.must_not for s in foreign)


def test_dataset_loader_rejects_bad_files(tmp_path: Path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="bad.jsonl:1"):
        load_dataset(bad)
    extra = tmp_path / "extra.jsonl"
    row = json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])
    row["expected"]["surprise"] = 1  # typo'd expectation keys must not be silently ignored
    extra.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset(extra)


def test_load_limit_and_subset():
    assert len(load_dataset(DATASET, limit=5)) == 5
    assert all("ci" in s.subset for s in load_dataset(DATASET, subset="ci"))


def sample(**over) -> Sample:
    base = dict(
        id="s-1", lang="en", type="policy", principal=Principal(user_id="u_100"),
        input="q", expected=Expected(route="policy", outcome="answered"),
    )  # fmt: skip
    return Sample(**(base | over))


def test_check_dataset_flags_problems():
    dup = [sample(id="a"), sample(id="a")]
    assert any("duplicate" in p for p in check_dataset(dup))
    assert any("unbalanced" in p for p in check_dataset([sample(id=str(i)) for i in range(4)]))
    empty = sample(id="e", expected=Expected())
    assert any("neither a route nor an outcome" in p for p in check_dataset([empty]))
    no_docs = [sample(id="p1"), sample(id="p2", lang="vi")]
    assert any("without expected docs" in p for p in check_dataset(no_docs))


# --- metrics --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answer,fact,ok",
    [
        ("Bạn có 7 ngày để đổi trả", "7 ngày", True),
        ("You have 7 DAYS", "7 days", True),
        ("Hoàn 350.000đ", "350000", True),  # thousands separators ignored
        ("Refund of 350,000 VND", "350000", True),
        ("Refund of 350 000", "350000", True),
        ("Điện thoại Nova", "dien thoai", True),  # diacritics ignored
        ("shipping in 2-3 days", ["2 to 3", "2-3"], True),  # any-of
        ("shipping in 5 days", ["2 to 3", "2-3"], False),
        ("7 days", "17 days", False),
        ("The order costs 3.5 million", "3500000", False),  # decimals are not thousands
    ],
)
def test_fact_matching(answer: str, fact, ok: bool):
    assert m.fact_present(answer, fact) is ok


def test_facts_satisfied_counts():
    assert m.facts_satisfied("a 7 days b momo", ["7 days", "momo", "zalopay"]) == (2, 3)
    assert m.facts_satisfied("anything", []) == (0, 0)


def test_forbidden_hits_are_case_and_accent_insensitive():
    assert m.forbidden_hits("Địa chỉ: TRẦN HƯNG ĐẠO", ["Trần Hưng Đạo", "nowhere"]) == [
        "Trần Hưng Đạo"
    ]
    assert m.forbidden_hits("clean", ["x"]) == []


@pytest.mark.parametrize(
    "answer,verdict",
    [
        ("Yes, your order is eligible for return.", "yes"),
        ("Unfortunately it is not eligible: the window expired.", "no"),
        ("Đơn của bạn đủ điều kiện đổi trả.", "yes"),
        ("Rất tiếc, đơn đã hết hạn đổi trả.", "no"),
        ("Đơn này không đủ điều kiện.", "no"),
        ("Máy lọc không khí của bạn vẫn còn trong thời hạn bảo hành đến 2028.", "yes"),
        ("Sản phẩm này không được bảo hành vì đã vào nước.", "no"),
        ("Bảo hành còn hiệu lực và sẽ hết hạn vào ngày 28/09/2028.", "yes"),
        ("Có, máy còn bảo hành đến 2028. Bảo hành không bao gồm hư hỏng do vào nước.", "yes"),
        ("Rất tiếc, đơn không đủ điều kiện. Bạn có thể liên hệ nhân viên để được hỗ trợ.", "no"),
        ("Thời hạn bảo hành đã hết hạn từ tháng trước.", "no"),
        ("Here is some information.", "unknown"),
    ],
)
def test_eligibility_polarity(answer: str, verdict: str):
    assert m.eligibility_polarity(answer) == verdict


def test_tools_ok():
    assert m.tools_ok(["get_order"], ["get_order", "get_shipment_status"])
    assert not m.tools_ok(["get_order", "check_stock"], ["get_order"])
    assert m.tools_ok([], [])
    assert not m.tools_ok([], ["get_order"])  # a policy question must not touch customer data


def test_language_ok():
    assert m.language_ok("Bạn có 7 ngày để đổi trả hàng", "vi")
    assert not m.language_ok("You have 7 days", "vi")


def test_ranking_metrics():
    items = [("a.md", "S1"), ("b.md", "1. Return window"), ("b.md", "S3")]
    assert m.first_rank(items, ["b.md"], ["Return window"]) == (2, 2)
    assert m.first_rank(items, ["b.md"], ["Nothing"]) == (2, None)
    assert m.first_rank(items, ["z.md"], []) == (None, None)
    assert m.hit_at_k(2, 3) and not m.hit_at_k(2, 1) and not m.hit_at_k(None, 6)
    assert m.reciprocal_rank(4) == 0.25 and m.reciprocal_rank(None) == 0.0
    assert m.mean([]) is None and m.mean([True, False]) == 0.5


def test_gate_curve_and_recommendation():
    pos, neg = [0.85, 0.88, 0.9], [0.7, 0.75]
    curve = m.gate_curve(pos, neg, [0.7, 0.8, 0.95])
    assert [(p.recall, p.specificity) for p in curve] == [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert m.recommend_threshold(pos, neg, curve) == 0.8  # midpoint of the clean gap
    # overlapping classes: choose the best balanced point instead
    overlap = m.gate_curve([0.8, 0.9], [0.85, 0.7], [0.75, 0.82, 0.95])
    assert m.recommend_threshold([0.8, 0.9], [0.85, 0.7], overlap) in (0.75, 0.82)
    assert m.recommend_threshold([], [0.5], curve) is None


def test_percentile():
    assert m.percentile([], 0.5) is None
    assert m.percentile([10, 20, 30, 40], 0.5) in (20, 30) and m.percentile([5], 0.95) == 5


# --- retrieval evaluation ---------------------------------------------------------------------------


@pytest.fixture
def indexed_retriever(store: VectorStore) -> Retriever:
    cfg = RetrievalConfig(top_k=6, score_threshold=0.25)
    ingest(
        ROOT / "knowledge",
        store=store,
        embeddings=HashingEmbeddings(),
        sparse=FakeSparse(),
        embedding_model="fake",
        cfg=cfg,
    )
    return Retriever(store, HashingEmbeddings(), cfg, FakeSparse())


def test_retrieval_evaluation_reports_ranking_and_gate(indexed_retriever: Retriever):
    samples = load_dataset(DATASET)
    r = evaluate_retrieval(indexed_retriever, samples, ks=[1, 3, 6], configured_threshold=0.3)
    assert r.n_evaluated >= 30
    assert r.hit_at[6] >= r.hit_at[3] >= r.hit_at[1]  # monotonic by construction
    assert 0 <= r.mrr <= 1 and set(r.by_lang) == {"en", "vi"}
    assert set(r.cross_lingual_hit_at) == {1, 3}
    assert r.cross_lingual_hit_at[3] >= r.cross_lingual_hit_at[1]
    g = r.gate
    assert g.n_positive >= 30 and g.n_negative >= 10
    assert g.curve and g.lowest_positives and g.highest_negatives
    assert all(
        a[1] <= b[1] for a, b in zip(g.lowest_positives, g.lowest_positives[1:], strict=False)
    )


def test_retrieval_evaluation_skips_samples_without_expectations(indexed_retriever: Retriever):
    only_personal = [s for s in load_dataset(DATASET) if s.type == "personal"]
    r = evaluate_retrieval(indexed_retriever, only_personal, ks=[1], configured_threshold=0.3)
    assert r.n_evaluated == 0 and r.mrr is None


def test_heuristic_routing_baseline_is_reported_per_type():
    r = evaluate_heuristic_routing(load_dataset(DATASET))
    assert r.source == "heuristic" and 0.5 < r.accuracy <= 1.0
    assert set(r.by_type) == {"policy", "personal", "combined", "trap"}
    assert r.n == len(load_dataset(DATASET))


# --- scoring a single sample ----------------------------------------------------------------------


def answer(**over) -> RagAnswer:
    base = dict(answer="x", language="en", route="policy", outcome="answered")
    return RagAnswer(**(base | over))


def test_score_sample_all_good():
    s = sample(
        expected=Expected(route="policy", outcome="answered", answer_facts=["7 days"], docs=["a"])
    )
    found = [{"doc_id": "a", "section": "s", "score": 0.9}]  # a policy answer must be grounded
    sc = score_sample(
        s, answer(answer="You have 7 days to return it, in English words.", retrieved=found), None
    )
    assert sc.answer_ok and sc.route_ok and sc.outcome_ok and sc.facts_ok and sc.tools_ok


def test_score_sample_detects_each_failure():
    exp = Expected(
        route="personal",
        outcome="answered",
        answer_facts=["GHTK"],
        tools=["get_order"],
        must_not=["secret"],
    )
    s = sample(type="personal", expected=exp, tags=["safety"])
    sc = score_sample(
        s, answer(answer="It is with a courier. secret", route="policy", outcome="no_info"), None
    )
    assert not sc.route_ok and not sc.outcome_ok and not sc.tools_ok
    assert sc.facts_found == 0 and sc.forbidden == ["secret"]
    assert not sc.answer_ok and sc.safety_ok is False


def test_business_correctness_checks_the_verdict_not_just_keywords():
    s = sample(type="combined", expected=Expected(route="combined", eligible=False))
    assert score_sample(s, answer(answer="Unfortunately it is not eligible."), None).polarity_ok
    wrong = score_sample(s, answer(answer="Yes, it is eligible."), None)
    assert wrong.polarity_ok is False and not wrong.answer_ok


# --- full pipeline run with a scripted model --------------------------------------------------------


@pytest.fixture
def rag_factory(indexed_retriever: Retriever, tool_client: DomainToolClient, app_config: AppConfig):
    def build(llm: SupportFakeLLM) -> SupportRAG:
        return SupportRAG(
            model=llm.model, retriever=indexed_retriever, tools=tool_client, config=app_config
        )

    return build


def subset(*ids: str) -> list[Sample]:
    wanted = {s.id: s for s in load_dataset(DATASET)}
    return [wanted[i] for i in ids]


def pick(*questions: str) -> list[Sample]:
    """Dataset samples chosen by their question text (ids are an implementation detail)."""
    wanted = {s.input: s for s in load_dataset(DATASET)}
    return [wanted[q] for q in questions]


async def test_pipeline_eval_scores_a_correct_system_perfectly(rag_factory):
    routes = {
        "Where is order #9999?": "personal",  # does not exist
        "Where is order #2001?": "personal",  # someone else's
        "What is the capital of France?": "out_of_scope",
        "Hello!": "chitchat",
    }
    samples = pick(*routes)

    class Oracle(SupportFakeLLM):
        def _respond(self, messages, tools):  # route according to the question
            question = str(messages[-1].content)
            self.route = routes.get(question, "policy")
            return super()._respond(messages, tools)

    results = await run_pipeline_eval(rag_factory(Oracle()), samples, run_name="t")
    report = summarise_pipeline(results)
    assert report.n == 4 and not report.errors and not report.failures, report.failures
    assert report.metrics["routing"] == 1.0 and report.metrics["answer"] == 1.0
    assert report.metrics["safety"] == 1.0 and report.metrics["language"] == 1.0
    assert report.usage.input_tokens > 0 and report.latency_ms_p50 is not None


async def test_pipeline_eval_exposes_a_system_that_leaks_or_misroutes(rag_factory):
    samples = pick("Where is order #2001?", "What is the capital of France?")
    llm = SupportFakeLLM(route="policy", answer="Order 2001 is VNP2001 shipped to Trần Hưng Đạo")
    results = await run_pipeline_eval(rag_factory(llm), samples, run_name="t")
    report = summarise_pipeline(results)
    assert report.metrics["routing"] == 0.0 and report.metrics["answer"] == 0.0
    assert report.failures and all("route" in str(f["problem"]) for f in report.failures)


async def test_a_crashing_sample_is_reported_not_fatal(rag_factory):
    samples = pick("Hello!", "What is the capital of France?")

    class Boom:
        calls = 0

        async def answer(self, *a, **k):
            Boom.calls += 1
            if Boom.calls == 1:
                raise RuntimeError("kaboom")
            return RagAnswer(answer="x", language="en", route="out_of_scope", outcome="refused")

    results = await run_pipeline_eval(Boom(), samples, run_name="t")  # type: ignore[arg-type]
    report = summarise_pipeline(results)
    assert report.n == 2 and len(report.errors) == 1 and "kaboom" in report.errors[0]
    assert any("crashed" in str(f["problem"]) for f in report.failures)


async def test_llm_judge_can_veto_and_its_own_failure_is_not_a_fail(rag_factory):
    s = subset("policy-en-001")[0]
    harsh = ScriptedChatModel(
        responder=lambda msgs, tools: ai_json({"correct": False, "reason": "vague"})
    )
    sc = score_sample(
        s, answer(answer="You have 7 days to return."), await LLMJudge(harsh).grade(s, "x")
    )
    assert sc.judge_ok is False and not sc.answer_ok

    def broken(msgs, tools):
        raise RuntimeError("judge down")

    verdict = await LLMJudge(ScriptedChatModel(responder=broken)).grade(s, "x")
    assert verdict.correct is None
    assert score_sample(
        s, answer(answer="You have 7 days to return."), verdict
    ).answer_ok  # unjudged, not wrong


async def test_judge_prompt_lists_expected_facts(rag_factory):
    s = subset("combined-en-001")[0]
    seen: list[str] = []

    def respond(msgs, tools):
        seen.append(str(msgs[-1].content))
        return ai_json({"correct": True, "reason": "ok"})

    await LLMJudge(ScriptedChatModel(responder=respond)).grade(s, "yes")
    assert "IS eligible for return" in seen[0] and s.input in seen[0]


# --- thresholds and reports -------------------------------------------------------------------------


def test_threshold_checks():
    checks = check_thresholds(
        {"routing": 0.95, "trajectory": 0.8, "answer": None, "business": 1.0, "safety": 0.99},
        EvalThresholds(),
    )
    verdicts = {c.name: c.passed for c in checks}
    assert verdicts == {
        "routing": True,
        "trajectory": False,
        "answer": None,
        "business": True,
        "safety": False,
    }


def make_report(checks) -> EvalReport:
    meta = EvalMeta(
        created_at="2026-10-07T00:00:00Z", mode="offline", dataset="d.jsonl", n_samples=3,
        embedding_model="e5", score_threshold=0.8, router="heuristic", run_name="unit",
    )  # fmt: skip
    return EvalReport(meta=meta, thresholds=checks)


def test_report_passes_unless_a_measured_metric_fails():
    ok = make_report(check_thresholds({"routing": 1.0}, EvalThresholds()))
    assert ok.passed  # unmeasured metrics do not fail an offline run
    bad = make_report(check_thresholds({"routing": 0.5}, EvalThresholds()))
    assert not bad.passed


async def test_report_renders_and_writes_files(
    tmp_path: Path, rag_factory, indexed_retriever: Retriever
):
    samples = [*pick("Hello!"), *subset("policy-en-001", "policy-vi-001")]
    llm = SupportFakeLLM(route="policy", answer="You have 7 days")
    results = await run_pipeline_eval(rag_factory(llm), samples, run_name="unit")
    pipeline = summarise_pipeline(results)
    retrieval = evaluate_retrieval(
        indexed_retriever, load_dataset(DATASET), ks=[1, 3], configured_threshold=0.3
    )
    report = EvalReport(
        meta=make_report([]).meta.model_copy(update={"mode": "full", "provider": "fake", "model": "m"}),
        retrieval=retrieval, pipeline=pipeline,
        thresholds=check_thresholds(pipeline.metrics, EvalThresholds()),
    )  # fmt: skip
    text = render_markdown(report)
    for heading in (
        "## Retrieval",
        "### Score-threshold calibration",
        "## Routing",
        "## Pipeline",
        "## Thresholds",
    ):
        assert heading in text
    assert "hit@1" in text and "Recommended threshold" in text and "Failures" in text
    md, js = write_report(report, tmp_path / "out", "unit")
    assert md.read_text(encoding="utf-8") == text
    assert json.loads(js.read_text(encoding="utf-8"))["meta"]["run_name"] == "unit"


def test_offline_report_says_what_it_does_not_measure():
    text = render_markdown(make_report(check_thresholds({"routing": 0.8}, EvalThresholds())))
    assert "Offline run" in text and "not measured" in text


# --- Langfuse (optional) ------------------------------------------------------------------------------


def test_tracing_is_a_noop_without_keys(settings):
    tracing = create_tracing(settings)
    assert not tracing.enabled
    assert tracing.trace_id("x") is None and tracing.handler(None) is None
    tracing.score("t", "n", 1.0)  # must not raise
    tracing.flush()


def test_tracing_forwards_to_the_client_when_enabled():
    calls: list[dict] = []

    class FakeClient:
        def create_trace_id(self, seed: str) -> str:
            return f"trace-{seed}"

        def create_score(self, **kw) -> None:
            calls.append(kw)

        def flush(self) -> None:
            calls.append({"flushed": True})

    t = Tracing(FakeClient())
    assert t.enabled and t.trace_id("run:s1") == "trace-run:s1"
    t.score("trace-1", "answer_ok", True, comment="c")
    t.score(None, "ignored", 1)  # no trace id: nothing is sent
    t.flush()
    assert calls == [
        {
            "name": "answer_ok",
            "value": 1.0,
            "trace_id": "trace-1",
            "data_type": "NUMERIC",
            "comment": "c",
        },
        {"flushed": True},
    ]


def test_tracing_enabled_by_keys_builds_a_real_handler(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")  # no network export from a unit test
    pytest.importorskip("langfuse.langchain")
    from support_agent.core.settings import Settings

    s = Settings(
        _env_file=None, app_config_path=ROOT / "config" / "app.yaml",
        langfuse_public_key="pk-lf-test", langfuse_secret_key="sk-lf-test", langfuse_host="http://localhost:9",
    )  # fmt: skip
    tracing = create_tracing(s)
    assert tracing.enabled
    trace_id = tracing.trace_id("run:s1")
    assert (
        trace_id == tracing.trace_id("run:s1") and len(trace_id) == 32
    )  # deterministic per sample
    assert tracing.handler(trace_id) is not None


async def test_eval_run_attaches_tracing_handler_and_scores(rag_factory):
    calls: list[tuple] = []

    class Recorder(Tracing):
        def __init__(self) -> None:
            super().__init__(object())

        def trace_id(self, seed: str) -> str:
            return f"t:{seed}"

        def handler(self, trace_id):
            return None

        def score(self, trace_id, name, value, *, comment=None) -> None:
            calls.append((trace_id, name, value))

        def flush(self) -> None:
            calls.append(("flush",))

    samples = pick("Hello!")
    llm = SupportFakeLLM(route="chitchat")
    results = await run_pipeline_eval(rag_factory(llm), samples, run_name="r", tracing=Recorder())
    trace = f"t:r:{samples[0].id}"
    assert results[0].answer.trace_id == trace
    assert (trace, "answer_ok", True) in calls and ("flush",) in calls


# --- regressions found by the first full run with a real model ---------------------------------


@pytest.mark.parametrize(
    "answer,lang,ok",
    [
        (
            "Your order contains Tai nghe Bluetooth BT20 and is in transit with the courier.",
            "en",
            True,
        ),
        ("Đơn hàng của bạn đang được giao bởi GHTK.", "vi", True),
        ("Your order is in transit.", "vi", False),
        ("Đơn hàng của bạn đang được giao.", "en", False),
        ("don hang cua ban dang duoc giao", "vi", True),  # Vietnamese typed without accents
        ("Hết hàng", "vi", True),
    ],
)
def test_answer_language_ignores_vietnamese_product_names_in_english_text(answer, lang, ok):
    assert m.language_ok(answer, lang) is ok


def test_judge_does_not_penalise_additional_accurate_detail():
    from support_agent.evals.judge import JUDGE_PROMPT

    assert "must NOT be penalised" in JUDGE_PROMPT and "minimum" in JUDGE_PROMPT


def test_injection_traps_reach_the_model_through_a_tool_that_returns_the_description():
    traps = [s for s in load_dataset(DATASET) if "indirect_injection" in s.tags]
    assert len(traps) >= 3
    # check_stock never returns the description, so those prompts would not test anything.
    assert all("search_products" in s.expected.tools for s in traps)


def test_amount_questions_are_not_scored_as_yes_no_verdicts():
    amount = [s for s in load_dataset(DATASET) if s.input.startswith(("If I return", "Nếu trả"))]
    assert amount and all(s.expected.eligible is None for s in amount)


def test_each_run_gets_its_own_tracing_session():
    from datetime import UTC, datetime

    from support_agent.evals.report import trace_session

    a = trace_session("baseline-full", datetime(2026, 10, 7, 17, 0, 0, tzinfo=UTC))
    b = trace_session("baseline-full", datetime(2026, 10, 7, 17, 5, 0, tzinfo=UTC))
    assert a == "baseline-full-20261007-170000" and a != b


# --- trajectory is judged the same for both engines ----------------------------------------------------


def test_policy_search_is_retrieval_not_a_data_tool():
    assert m.data_tools(["search_policy", "get_order", "search_policy"]) == ["get_order"]
    assert m.data_tools([]) == []


def _policy_sample(**exp) -> Sample:
    return sample(
        expected=Expected(route="policy", outcome="answered", docs=["return-policy.en.md"], **exp)
    )


def test_a_policy_question_may_search_policy_but_must_actually_retrieve():
    s = _policy_sample()
    searched = answer(
        tool_calls=["search_policy"],
        retrieved=[{"doc_id": "return-policy.en.md", "section": "1", "score": 0.9}],
    )
    assert score_sample(s, searched, None).tools_ok  # the agent's way: a tool, with a result
    pipeline_style = answer(
        tool_calls=[], retrieved=[{"doc_id": "return-policy.en.md", "section": "1", "score": 0.9}]
    )
    assert score_sample(s, pipeline_style, None).tools_ok  # the Phase A way: same verdict
    assert not score_sample(
        s, answer(tool_calls=[], retrieved=[]), None
    ).tools_ok  # answered unaided


def test_touching_customer_data_is_still_wrong_for_a_policy_question():
    s = _policy_sample()
    leaky = answer(
        tool_calls=["get_order"], retrieved=[{"doc_id": "x", "section": "s", "score": 1.0}]
    )
    assert not score_sample(s, leaky, None).tools_ok


def test_a_question_that_needs_no_documents_may_still_look_them_up():
    s = sample(expected=Expected(route="policy", outcome="no_info"))  # e.g. "do you gift-wrap?"
    assert score_sample(s, answer(tool_calls=["search_policy"], retrieved=[]), None).tools_ok


def test_return_questions_need_the_rules_engine_not_a_specific_prefetch():
    returns = [
        s
        for s in load_dataset(DATASET)
        if s.expected.eligible is not None or "return" in s.input.lower()
    ]
    needing = [s for s in returns if "check_return_eligibility" in s.expected.tools]
    assert needing and all(s.expected.tools == ["check_return_eligibility"] for s in needing)


# --- saved runs can be graded again without a model --------------------------------------------------------


def _record(sample_id: str, **over) -> dict:
    base = dict(answer="x", language="en", route="policy", outcome="answered")
    return {"id": sample_id, "answer": base | over, "verdict": None, "error": None}


def test_rescoring_changes_the_grade_but_never_the_answers():
    from support_agent.evals.runner import rescore_pipeline

    strict = sample(
        id="q1", expected=Expected(route="policy", outcome="answered", answer_facts=["7 days"])
    )
    lenient = strict.model_copy(update={"expected": Expected(route="policy", outcome="answered")})
    saved = {"pipeline": {"records": [_record("q1", answer="Seven days")]}}

    assert rescore_pipeline(saved, [strict]).metrics["answer"] == 0.0  # the fact is missing
    assert (
        rescore_pipeline(saved, [lenient]).metrics["answer"] == 1.0
    )  # same answer, new expectation
    assert saved["pipeline"]["records"][0]["answer"]["answer"] == "Seven days"


def test_rescoring_ignores_questions_no_longer_in_the_dataset_and_keeps_crashes():
    from support_agent.evals.runner import rescore_pipeline

    kept = sample(id="q1")
    crashed = sample(id="q2")
    saved = {
        "pipeline": {
            "records": [
                _record("q1"),
                {"id": "q2", "answer": None, "verdict": None, "error": "boom"},
                _record("gone"),
            ]
        }
    }
    report = rescore_pipeline(saved, [kept, crashed])
    assert report.n == 2 and report.errors == ["q2: boom"]


async def test_a_run_stores_one_record_per_question(rag_factory):
    samples = pick("Hello!", "What is the capital of France?")
    llm = SupportFakeLLM(route="chitchat")
    report = summarise_pipeline(await run_pipeline_eval(rag_factory(llm), samples, run_name="t"))
    assert [r.id for r in report.records] == [s.id for s in samples]
    assert all(r.answer is not None and r.answer.outcome == "refused" for r in report.records)
    assert json.loads(report.model_dump_json())["records"][0]["answer"]["answer"]  # serialisable


# --- request flows ------------------------------------------------------------------------------


def _draft_ok(**kw):
    from support_agent.evals import metrics

    base = dict(
        expected_type="refund", amount=None, priority=None, contains=[], decision=None,
        interrupt={"draft_type": "refund", "priority_review": False, "summary": {"amount": 350000}},
        created=[],
    )  # fmt: skip
    return metrics.draft_ok(**{**base, **kw})


def test_draft_ok_reads_the_confirmation_and_the_stored_drafts():
    assert _draft_ok(amount=350000, priority=False)
    assert not _draft_ok(amount=350001)
    assert not _draft_ok(priority=True)
    assert not _draft_ok(expected_type="warranty")
    assert not _draft_ok(interrupt=None)  # nothing was shown
    assert _draft_ok(contains=["350000"]) and not _draft_ok(contains=["dead pixels"])


def test_draft_ok_checks_what_was_stored_for_each_decision():
    made = [{"type": "refund", "status": "pending"}]
    assert _draft_ok(decision="approve", created=made)
    assert not _draft_ok(decision="approve", created=[])  # approved but nothing stored
    assert not _draft_ok(decision="approve", created=made * 2)
    assert _draft_ok(decision="reject") and not _draft_ok(decision="reject", created=made)
    assert _draft_ok() and not _draft_ok(created=made)  # no answer given: nothing may exist


def test_a_sample_that_expects_no_request_fails_if_one_was_proposed_or_stored():
    assert _draft_ok(expected_type=None, interrupt=None, created=[])
    assert not _draft_ok(expected_type=None)
    assert not _draft_ok(expected_type=None, interrupt=None, created=[{"type": "refund"}])


def test_two_approvals_of_the_same_refund_are_reported_by_the_dataset_check():
    from support_agent.evals.dataset import Sample, check_dataset

    def approve(id_: str, lang: str) -> Sample:
        return Sample.model_validate(
            {
                "id": id_, "lang": lang, "type": "aftersales", "principal": {"user_id": "u_100"},
                "input": "refund", "expected": {"route": "personal", "draft_type": "refund",
                                                 "decision": "approve"},
            }
        )  # fmt: skip

    problems = check_dataset([approve("a", "en"), approve("b", "vi")])
    assert any("approves what a already approves" in p for p in problems)
