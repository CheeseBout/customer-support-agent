from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from support_agent.cli import app
from support_agent.core.settings import _read_app_yaml, reset_settings_cache
from tests.conftest import DEMO, DEMO_APP, ROOT
from tests.fakes import AgentFakeLLM, FakeSparse, HashingEmbeddings, SupportFakeLLM

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway deployment: SQLite shop DB, embedded Qdrant, SQLite mapping, fake models."""
    cfg = _read_app_yaml(DEMO_APP)  # the shared defaults plus the demo shop's own rules
    cfg["mapping"] = {"path": str(DEMO / "config" / "schema_mapping.sqlite.yaml")}
    cfg["retrieval"]["score_threshold"] = 0.25
    cfg["knowledge"] = {"dir": str(DEMO / "knowledge")}
    cfg["evals"]["dataset"] = str(DEMO / "evals" / "datasets" / "baseline.jsonl")
    cfg["evals"]["report_dir"] = str(tmp_path / "evals" / "reports")
    app_yaml = tmp_path / "app.yaml"
    app_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_CONFIG_PATH", str(app_yaml))
    monkeypatch.setenv("BUSINESS_DB_TYPE", "sqlite")
    monkeypatch.setenv(
        "BUSINESS_DB_URL", f"sqlite+aiosqlite:///{(tmp_path / 'shop.db').as_posix()}"
    )
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("MCP_PRINCIPAL_SECRET", "cli-test-secret")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # offline embedding models
    monkeypatch.setattr(
        "support_agent.llm.factory.get_embeddings", lambda s=None: HashingEmbeddings()
    )
    monkeypatch.setattr("support_agent.runtime.get_embeddings", lambda s=None: HashingEmbeddings())
    monkeypatch.setattr("support_agent.rag.sparse.SparseEncoder", lambda *a, **k: FakeSparse())
    monkeypatch.setattr("support_agent.runtime.SparseEncoder", lambda *a, **k: FakeSparse())
    reset_settings_cache()
    yield tmp_path
    reset_settings_cache()


def invoke(*args: str):
    return runner.invoke(app, list(args), catch_exceptions=False)


def test_init_creates_env_file_and_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(ROOT / ".env.example", tmp_path / ".env.example")
    result = invoke("init")
    assert result.exit_code == 0 and (tmp_path / ".env").exists()
    assert all((tmp_path / d).is_dir() for d in ("data", "knowledge", "config"))
    again = invoke("init")
    assert "already exists" in again.output


def test_init_fails_outside_a_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == 1


def test_seed_demo_then_validate_mapping(env: Path):
    seeded = invoke("seed-demo")
    assert seeded.exit_code == 0 and '"orders": 10' in seeded.output
    ok = invoke("validate-mapping")
    assert ok.exit_code == 0 and "Mapping OK" in ok.output


def test_validate_mapping_fails_on_a_database_that_does_not_match(env: Path):
    result = runner.invoke(app, ["validate-mapping"])  # nothing seeded: tables are missing
    assert result.exit_code == 1 and "does not exist" in result.output


def test_validate_mapping_reports_a_dialect_mismatch(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BUSINESS_DB_TYPE", "postgres")
    reset_settings_cache()
    result = runner.invoke(app, ["validate-mapping"])
    assert result.exit_code == 1 and "dialect" in result.output


def test_seed_demo_needs_a_url(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BUSINESS_DB_URL")
    reset_settings_cache()
    assert runner.invoke(app, ["seed-demo"]).exit_code == 1


def test_ingest_is_incremental_from_the_cli(env: Path):
    first = invoke("ingest")
    assert first.exit_code == 0 and "added=8" in first.output
    second = invoke("ingest")
    assert "unchanged=8" in second.output and "added=0" in second.output
    full = invoke("ingest", "--full")
    assert "added=8" in full.output


def test_ingest_reports_a_missing_directory(env: Path):
    result = runner.invoke(app, ["ingest", "--dir", str(env / "nope")])
    assert result.exit_code != 0


def test_check_reports_each_component(env: Path):
    invoke("seed-demo")
    invoke("ingest")
    result = invoke("check", "--skip-llm")
    assert result.exit_code == 0
    for word in ("config", "qdrant", "database", "llm", "skip"):
        assert word in result.output


def test_check_fails_when_the_index_is_empty(env: Path):
    invoke("seed-demo")
    result = runner.invoke(app, ["check", "--skip-llm"])
    assert result.exit_code == 1 and "ingest" in result.output


def _use_fake_llm(monkeypatch: pytest.MonkeyPatch, fake: SupportFakeLLM) -> None:
    monkeypatch.setattr("support_agent.runtime.get_chat_model", lambda **kw: fake.model)


def test_pipeline_engine_end_to_end_through_a_real_mcp_subprocess(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    """CLI -> pipeline -> MCP client -> spawned `python -m ...mcp_db.server` -> SQLite."""
    invoke("seed-demo")
    invoke("ingest")
    _use_fake_llm(monkeypatch, SupportFakeLLM(route="personal", answer="In transit with GHTK."))

    mine = json.loads(
        invoke(
            "ask", "Where is order #1236?", "--user", "u_100", "--json", "--engine", "pipeline"
        ).output
    )
    assert mine["outcome"] == "answered" and mine["answer"] == "In transit with GHTK."
    assert mine["tool_calls"] == ["get_order", "get_shipment_status"]

    theirs = json.loads(
        invoke(
            "ask", "Where is order #2001?", "--user", "u_100", "--json", "--engine", "pipeline"
        ).output
    )
    assert theirs["outcome"] == "not_found" and theirs["tool_calls"] == ["get_order"]

    owner = json.loads(
        invoke(
            "ask", "Where is order #2001?", "--user", "u_101", "--json", "--engine", "pipeline"
        ).output
    )
    assert owner["outcome"] == "answered"


def test_pipeline_engine_prints_answer_and_sources(env: Path, monkeypatch: pytest.MonkeyPatch):
    invoke("ingest")
    _use_fake_llm(monkeypatch, SupportFakeLLM(route="policy", answer="Seven days."))
    result = invoke(
        "ask", "return window days within delivery", "--no-tools", "--engine", "pipeline"
    )
    assert "Seven days." in result.output and "Sources:" in result.output
    assert "return-policy" in result.output


def test_ask_rejects_an_unknown_role(env: Path):
    assert runner.invoke(app, ["ask", "hi", "--role", "admin"]).exit_code != 0


def test_eval_offline_writes_reports_and_calibrates_the_gate(env: Path):
    invoke("ingest")
    result = invoke("eval", "--offline")
    assert result.exit_code == 0
    for text in ("Evaluation report", "Score-threshold calibration", "Routing", "not measured"):
        assert text in result.output
    reports = env / "evals" / "reports"
    # report_dir comes from config (relative to the working directory)
    md = reports / "baseline-offline.md"
    assert md.exists() and "Recommended threshold" in md.read_text(encoding="utf-8")
    payload = json.loads((reports / "baseline-offline.json").read_text(encoding="utf-8"))
    assert payload["meta"]["mode"] == "offline" and payload["retrieval"]["gate"]["curve"]


def test_eval_fail_under_exits_nonzero_when_a_metric_is_below_threshold(env: Path):
    invoke("ingest")
    # the keyword router cannot reach 92% routing accuracy on the dataset
    result = runner.invoke(app, ["eval", "--offline", "--fail-under"])
    assert result.exit_code == 1 and "Below threshold" in result.output


def test_eval_subset_and_limit_name_the_report_accordingly(env: Path):
    invoke("ingest")
    invoke("eval", "--offline", "--subset", "ci")
    assert (env / "evals" / "reports" / "offline-ci.md").exists()
    invoke("eval", "--offline", "--limit", "6")
    assert (env / "evals" / "reports" / "offline-partial.md").exists()


def test_eval_rejects_missing_dataset_and_empty_selection(env: Path):
    missing = runner.invoke(app, ["eval", "--offline", "--dataset", str(env / "nope.jsonl")])
    assert missing.exit_code == 1 and "not found" in missing.output
    empty = runner.invoke(app, ["eval", "--offline", "--subset", "does-not-exist"])
    assert empty.exit_code == 1 and "No samples" in empty.output


# --- the agent engine (default) ----------------------------------------------------------------


def _use_agent_llm(monkeypatch: pytest.MonkeyPatch, fake: AgentFakeLLM) -> None:
    monkeypatch.setattr("support_agent.runtime.get_chat_model", lambda **kw: fake.model)


def test_agent_answers_through_a_real_mcp_subprocess_and_isolates_customers(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    invoke("seed-demo")
    invoke("ingest")
    llm = AgentFakeLLM(
        [
            {
                "tools": [
                    ("get_order", {"order_id": "1236"}),
                    ("get_shipment_status", {"order_id": "1236"}),
                ]
            },
            {"text": "In transit with GHTK."},
            {"tools": [("get_order", {"order_id": "2001"})]},
            {"text": "I could not find order 2001."},
            {"tools": [("get_order", {"order_id": "2001"})]},
            {"text": "It holds a phone."},
        ],
        route="personal",
    )
    _use_agent_llm(monkeypatch, llm)

    mine = json.loads(invoke("ask", "Where is order #1236?", "--user", "u_100", "--json").output)
    assert mine["outcome"] == "answered" and mine["answer"] == "In transit with GHTK."
    assert mine["tool_calls"] == ["get_order", "get_shipment_status"] and mine["session_id"]

    theirs = json.loads(invoke("ask", "Where is order #2001?", "--user", "u_100", "--json").output)
    assert theirs["outcome"] == "not_found"
    owner = json.loads(invoke("ask", "Where is order #2001?", "--user", "u_101", "--json").output)
    assert owner["outcome"] == "answered"


def test_agent_streams_to_the_console_and_continues_a_saved_session(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    invoke("ingest")
    llm = AgentFakeLLM(
        [
            {"tools": [("search_policy", {"query": "return window days delivery"})]},
            {"text": "Seven days [D1]."},
            {"tools": [("search_policy", {"query": "return window days delivery"})]},
            {"text": "Still seven days [D1]."},
        ],
        route="policy",
    )
    _use_agent_llm(monkeypatch, llm)
    first = invoke("ask", "return window days within delivery", "--no-tools")
    # no data tools were requested, but the policy tool needs none: the answer streams in
    assert "Seven days." in first.output and "[D1]" not in first.output
    assert "Sources:" in first.output and "session=" in first.output
    sid = first.output.split("session=")[1].split("[")[0].split()[0]

    second = invoke("ask", "and again?", "--no-tools", "--session", sid)
    assert "Still seven days." in second.output and sid in second.output
    # the second turn's model call carried the first exchange
    carried = " ".join(str(m.content) for m in llm.agent_calls()[-1])
    assert "return window days within delivery" in carried


def test_chat_runs_several_turns_in_one_session(env: Path, monkeypatch: pytest.MonkeyPatch):
    llm = AgentFakeLLM([], route="chitchat")
    _use_agent_llm(monkeypatch, llm)
    result = runner.invoke(app, ["chat", "--no-tools"], input="hello\nthanks\n/exit\n")
    assert result.exit_code == 0
    assert result.output.count("How can I help") == 2
    assert "--session" in result.output  # tells the user how to resume


def test_ask_rejects_an_unknown_engine(env: Path):
    assert runner.invoke(app, ["ask", "hi", "--engine", "nope"]).exit_code != 0


def test_eval_with_the_agent_engine_compares_against_the_baseline(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    invoke("ingest")
    baseline = DEMO / "evals" / "reports" / "baseline-full.json"
    reports = env / "evals" / "reports"
    reports.mkdir(parents=True)
    (reports / "baseline-full.json").write_text(
        baseline.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _use_agent_llm(monkeypatch, AgentFakeLLM([], route="chitchat"))

    result = invoke("eval", "--limit", "4")
    md = (reports / "agent-partial.md").read_text(encoding="utf-8")
    assert "Engine | agent (prompt v9)" in md
    assert "Comparison with `baseline-full`" in md
    payload = json.loads((reports / "agent-partial.json").read_text(encoding="utf-8"))
    assert (
        payload["meta"]["engine"] == "agent" and payload["comparison"]["against"] == "baseline-full"
    )
    assert "Report written" in result.output


# --- regrading a saved run ---------------------------------------------------------------------


def test_rescore_regrades_a_saved_run_without_calling_any_model(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    invoke("ingest")
    reports = env / "evals" / "reports"
    reports.mkdir(parents=True)
    _use_agent_llm(monkeypatch, AgentFakeLLM([], route="chitchat"))
    invoke("eval", "--limit", "4")
    saved = reports / "agent-partial.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["pipeline"]["records"]

    def no_model(**kw):
        raise AssertionError("rescoring must not call a model")

    monkeypatch.setattr("support_agent.runtime.get_chat_model", no_model)
    monkeypatch.setattr("support_agent.llm.factory.get_chat_model", no_model)
    result = invoke("eval", "--rescore", str(saved), "--limit", "4")

    md = (reports / "agent-partial-rescored.md").read_text(encoding="utf-8")
    assert "Rescored from | `agent-partial` (no model was called)" in md
    payload = json.loads((reports / "agent-partial-rescored.json").read_text(encoding="utf-8"))
    assert payload["meta"]["rescored_from"] == "agent-partial"
    assert payload["pipeline"]["records"] and "Report written" in result.output


def test_rescore_rejects_a_missing_report_and_one_without_records(env: Path):
    missing = runner.invoke(app, ["eval", "--rescore", str(env / "nope.json")])
    assert missing.exit_code == 1 and "not found" in missing.output
    old = DEMO / "evals" / "reports" / "baseline-offline.json"  # an offline report has no records
    result = runner.invoke(app, ["eval", "--rescore", str(old)])
    assert result.exit_code == 1 and "no per-question records" in result.output


# --- staff review of customer requests ------------------------------------------------------------


def test_staff_can_review_the_queue_from_the_command_line(env: Path, monkeypatch):
    import asyncio

    from support_agent.core.principal import Principal
    from support_agent.drafts.repository import create_repository
    from support_agent.drafts.service import DraftService

    url = f"sqlite+aiosqlite:///{(env / 'drafts.db').as_posix()}"
    monkeypatch.setenv("DRAFTS_DB_URL", url)
    reset_settings_cache()
    assert invoke("drafts", "init").exit_code == 0

    async def make() -> str:
        repo = create_repository(url)
        draft, _ = await DraftService(repo, get_rules()).create(
            Principal(user_id="u_100"),
            "warranty",
            {"order_id": "1234", "sku": "EAR-BT20", "issue_description": "battery drains"},
            "s1",
        )
        await repo.close()
        return draft.id

    def get_rules():
        from support_agent.core.settings import get_settings

        return get_settings().app.business_rules

    draft_id = asyncio.run(make())
    listing = invoke("drafts", "list")
    assert draft_id in listing.output
    assert runner.invoke(app, ["drafts", "reject", draft_id, "--note", ""]).exit_code == 1
    assert invoke("drafts", "approve", draft_id, "--staff", "s_9").exit_code == 0
    assert "Nothing in the queue" in invoke("drafts", "list").output
    assert '"status":"approved"' in invoke("drafts", "show", draft_id).output.replace(" ", "")


def test_serve_refuses_to_start_without_a_jwt_secret(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JWT_SECRET", raising=False)
    reset_settings_cache()
    result = runner.invoke(app, ["serve"])
    assert result.exit_code == 1 and "JWT_SECRET" in result.output


def test_serve_starts_uvicorn_with_the_app(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JWT_SECRET", "s" * 40)
    reset_settings_cache()
    started: dict[str, object] = {}
    monkeypatch.setattr(
        "uvicorn.run", lambda application, **kw: started.update(app=application, **kw)
    )
    result = invoke("serve", "--host", "0.0.0.0", "--port", "9100")
    assert result.exit_code == 0
    assert started["host"] == "0.0.0.0" and started["port"] == 9100
    assert started["app"].title == "Customer Support Agent"  # type: ignore[attr-defined]


# --- calibrate -------------------------------------------------------------------------------------------------


def test_calibrate_reports_a_threshold_for_the_demo_documents(env: Path):
    invoke("ingest")
    result = invoke("calibrate", str(DEMO / "evals" / "calibration.yaml"))
    assert "Recommended threshold" in result.output and "score_threshold:" in result.output


def test_calibrate_needs_an_index(env: Path):
    result = runner.invoke(app, ["calibrate", str(DEMO / "evals" / "calibration.yaml")])
    assert result.exit_code == 1 and "ingest" in result.output


def test_calibrate_reports_a_missing_questions_file(env: Path):
    result = runner.invoke(app, ["calibrate", str(env / "nope.yaml")])
    assert result.exit_code == 1 and "not found" in result.output


# --- introspect-db -------------------------------------------------------------------------------------------------------


def test_introspect_db_prints_a_draft_and_what_it_found(env: Path):
    invoke("seed-demo")
    result = invoke("introspect-db")
    assert "table: order_lines" in result.output and "dialect: sqlite" in result.output
    assert "validate-mapping" in result.output and "--write" in result.output


def test_introspect_db_writes_a_draft_the_validator_accepts(env: Path):
    invoke("seed-demo")
    target = env / "config" / "draft.yaml"
    result = invoke("introspect-db", "--write", str(target))
    assert target.exists() and "Draft written" in result.output
    from support_agent.mcp_db.mapping import load_mapping

    assert set(load_mapping(target).entities) >= {"customer", "order", "order_item", "shipment"}
    monkey_cfg = yaml.safe_load((env / "app.yaml").read_text(encoding="utf-8"))
    monkey_cfg["mapping"] = {"path": str(target)}
    (env / "app.yaml").write_text(yaml.safe_dump(monkey_cfg, allow_unicode=True), encoding="utf-8")
    reset_settings_cache()
    assert "Mapping OK" in invoke("validate-mapping").output


def test_introspect_db_does_not_overwrite_an_edited_file_without_force(env: Path):
    invoke("seed-demo")
    target = env / "mine.yaml"
    target.write_text("dialect: sqlite\nentities: {}\n", encoding="utf-8")
    refused = runner.invoke(app, ["introspect-db", "--write", str(target)])
    assert refused.exit_code == 1 and "--force" in refused.output
    assert target.read_text(encoding="utf-8").startswith("dialect: sqlite\nentities: {}")
    invoke("introspect-db", "--write", str(target), "--force")
    assert "order_lines" in target.read_text(encoding="utf-8")


def test_introspect_db_replaces_the_untouched_template_without_force(env: Path):
    invoke("seed-demo")
    target = env / "schema_mapping.yaml"
    shutil.copy(ROOT / "config" / "schema_mapping.yaml", target)
    invoke("introspect-db", "--write", str(target))
    assert "order_lines" in target.read_text(encoding="utf-8")


def test_introspect_db_needs_a_database_url(env: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BUSINESS_DB_URL")
    reset_settings_cache()
    result = runner.invoke(app, ["introspect-db"])
    assert result.exit_code == 1 and "BUSINESS_DB_URL" in result.output


def test_introspect_db_reports_an_unreachable_database_without_a_traceback(
    env: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "BUSINESS_DB_URL", "sqlite+aiosqlite:///" + (env / "no" / "such.db").as_posix()
    )
    reset_settings_cache()
    result = runner.invoke(app, ["introspect-db"])
    assert result.exit_code == 1 and "Could not read the database" in result.output
