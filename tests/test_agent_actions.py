"""Requests the customer confirms: propose -> stop for the customer -> create (SPEC 11.4, 14.4)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver

from support_agent.agent.agent import SupportAgent
from support_agent.agent.drafting import DraftProposer
from support_agent.agent.events import AgentEvent
from support_agent.agent.graph import AgentDeps, build_graph
from support_agent.core.settings import AppConfig
from support_agent.drafts.repository import SqlDraftRepository
from support_agent.drafts.service import DraftService
from support_agent.memory.workspace import Workspace
from support_agent.tools.client import DomainToolClient
from tests.conftest import ALICE, BOB
from tests.fakes import AgentFakeLLM
from tests.test_agent import collect, done, kinds, tokens
from tests.test_agent import fast_config as fast_config
from tests.test_agent import retriever as retriever

NOW = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)

REFUND = (
    "propose_draft",
    {
        "draft_type": "refund",
        "order_id": "1234",
        "reason_code": "defective",
        "reason_text": "left bud is dead",
    },
)
WARRANTY = (
    "propose_draft",
    {
        "draft_type": "warranty",
        "order_id": "1234",
        "sku": "EAR-BT20",
        "issue_description": "battery drains fast",
    },
)
ORDER = (
    "propose_draft",
    {
        "draft_type": "order",
        "items": [{"sku": "PHN-X100", "qty": 1}],
        "shipping_address": "12 Le Loi, District 1",
        "payment_method": "bank_transfer",
    },
)


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@dataclass
class Rig:
    build: Callable[..., SupportAgent]
    drafts: DraftService
    clock: Clock


@pytest_asyncio.fixture
async def rig(
    retriever: Any,  # noqa: F811
    tool_client: DomainToolClient,
    fast_config: AppConfig,  # noqa: F811
    tmp_path: Path,
) -> AsyncIterator[Rig]:
    repo = SqlDraftRepository.from_url(f"sqlite+aiosqlite:///{(tmp_path / 'drafts.db').as_posix()}")
    await repo.create_schema()
    clock = Clock()
    drafts = DraftService(repo, fast_config.business_rules)
    proposer = DraftProposer(tool_client, drafts, fast_config.business_rules)

    def build(
        llm: AgentFakeLLM,
        *,
        with_drafts: bool = True,
        checkpointer: Any = None,
        config: AppConfig | None = None,
        workspace: Workspace | None = None,
    ) -> SupportAgent:
        cfg = config or fast_config
        deps = AgentDeps(
            model=llm.model,
            retriever=retriever,
            client=tool_client,
            config=cfg,
            drafts=drafts if with_drafts else None,
            proposer=proposer if with_drafts else None,
            workspace=workspace,
            clock=clock,
        )
        graph = build_graph(deps, checkpointer or InMemorySaver())
        return SupportAgent(
            graph=graph, retriever=retriever, config=cfg, drafts=drafts, workspace=workspace
        )

    yield Rig(build, drafts, clock)
    await repo.close()


def llm_for(*steps: dict[str, Any]) -> AgentFakeLLM:
    return AgentFakeLLM(list(steps), route="combined")


def waiting(events: list[AgentEvent]) -> dict[str, Any]:
    assert events[-1].kind == "interrupt", kinds(events)
    return events[-1].data


async def ask(agent: SupportAgent, text: str, session: str = "s1", who=ALICE) -> list[AgentEvent]:
    return await collect(agent.stream(text, who, session_id=session))


async def answer_with(
    agent: SupportAgent, stop: dict[str, Any], decision: str, who=ALICE, **edits: Any
) -> list[AgentEvent]:
    return await collect(
        agent.resume(who, stop["session_id"], stop["id"], decision, edits=edits or None)
    )


# --- the happy paths ------------------------------------------------------------------------------


async def test_a_refund_stops_for_the_customer_and_nothing_is_created_yet(rig: Rig):
    agent = rig.build(llm_for({"tools": [REFUND]}))
    events = await ask(agent, "My earbuds are dead, I want a refund for order 1234")
    stop = waiting(events)
    assert "done" not in kinds(events)
    assert stop["kind"] == "confirm_draft" and stop["draft_type"] == "refund"
    assert stop["allowed_decisions"] == ["approve", "reject", "edit"]
    assert stop["editable_fields"] == ["reason_code", "reason_text"]
    assert stop["summary"]["order_id"] == "1234" and stop["summary"]["amount"] > 0
    assert await rig.drafts.list_for(ALICE) == []  # asking is not submitting


async def test_approving_creates_one_pending_draft_and_the_model_reports_it(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "Your refund request was submitted."})
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "refund order 1234 please"))
    events = await answer_with(agent, stop, "approve")
    result = done(events)
    created = await rig.drafts.list_for(ALICE)
    assert len(created) == 1 and created[0].status == "pending" and created[0].type == "refund"
    assert result["drafts"] == [
        {
            "id": created[0].id,
            "type": "refund",
            "status": "pending",
            "priority_review": created[0].priority_review,
        }
    ]
    assert tokens(events) == "Your refund request was submitted."
    seen = "\n".join(str(m.content) for m in llm.agent_calls()[-1] if m.type == "tool")
    assert '"status": "created"' in seen and created[0].id in seen


async def test_the_amount_comes_from_the_database_not_from_the_model(rig: Rig):
    args = REFUND[1] | {"refundable_amount": 99_999_999}  # not a field: must be refused or ignored
    agent = rig.build(llm_for({"tools": [("propose_draft", args)]}, {"text": "Sorry."}))
    events = await ask(agent, "refund order 1234")
    assert not rig.drafts or await rig.drafts.list_for(ALICE) == []
    if events[-1].kind == "interrupt":
        await answer_with(agent, events[-1].data, "approve")
        (draft,) = await rig.drafts.list_for(ALICE)
        assert draft.payload["refundable_amount"] != 99_999_999


async def test_declining_creates_nothing(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "No problem, I did not submit it."})
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "refund order 1234"))
    result = done(await answer_with(agent, stop, "reject"))
    assert await rig.drafts.list_for(ALICE) == [] and result["drafts"] == []
    seen = "\n".join(str(m.content) for m in llm.agent_calls()[-1] if m.type == "tool")
    assert "declined_by_customer" in seen


async def test_warranty_and_order_requests_use_the_same_flow(rig: Rig):
    llm = llm_for(
        {"tools": [WARRANTY]},
        {"text": "Submitted."},
        {"tools": [ORDER]},
        {"text": "Order placed for review."},
    )
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "my earbuds battery drains fast, warranty please", "w"))
    assert stop["draft_type"] == "warranty" and stop["editable_fields"] == ["issue_description"]
    done(await answer_with(agent, stop, "approve"))
    stop = waiting(await ask(agent, "I want to buy the Nova X100", "o"))
    assert stop["draft_type"] == "order"
    assert sorted(stop["editable_fields"]) == ["items", "payment_method", "shipping_address"]
    done(await answer_with(agent, stop, "approve"))
    assert sorted(d.type for d in await rig.drafts.list_for(ALICE)) == ["order", "warranty"]


# --- editing ---------------------------------------------------------------------------------------


async def test_editing_asks_again_with_a_new_confirmation_id(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "Submitted."})
    agent = rig.build(llm)
    first = waiting(await ask(agent, "refund order 1234"))
    second = waiting(await answer_with(agent, first, "edit", reason_text="screen cracked"))
    assert second["id"] != first["id"] and second["kind"] == "confirm_draft"
    done(await answer_with(agent, second, "approve"))
    (draft,) = await rig.drafts.list_for(ALICE)
    assert draft.payload["reason_text"] == "screen cracked"


async def test_a_field_that_may_not_be_edited_is_refused(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "That cannot be changed."})
    agent = rig.build(llm)
    first = waiting(await ask(agent, "refund order 1234"))
    events = await answer_with(agent, first, "edit", refundable_amount=1)
    done(events)  # no new confirmation: the edit was rejected and the model explained
    assert await rig.drafts.list_for(ALICE) == []


async def test_the_old_confirmation_stops_working_after_an_edit(rig: Rig):
    agent = rig.build(llm_for({"tools": [REFUND]}, {"text": "ok"}))
    first = waiting(await ask(agent, "refund order 1234"))
    waiting(await answer_with(agent, first, "edit", reason_text="changed"))
    stale = await answer_with(agent, first, "approve")
    assert stale[-1].kind == "error" and stale[-1].data["code"] == "NO_PENDING_CONFIRMATION"
    assert await rig.drafts.list_for(ALICE) == []


# --- time, staleness, abandonment ------------------------------------------------------------------


async def test_an_expired_confirmation_creates_nothing(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "That timed out; shall I try again?"})
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "refund order 1234"))
    rig.clock.now += timedelta(minutes=16)
    done(await answer_with(agent, stop, "approve"))
    assert await rig.drafts.list_for(ALICE) == []
    seen = "\n".join(str(m.content) for m in llm.agent_calls()[-1] if m.type == "tool")
    assert "expired" in seen


async def test_a_wrong_confirmation_id_changes_nothing(rig: Rig):
    agent = rig.build(llm_for({"tools": [REFUND]}))
    stop = waiting(await ask(agent, "refund order 1234"))
    events = await collect(agent.resume(ALICE, stop["session_id"], "not-the-id", "approve"))
    assert events[-1].kind == "error" and events[-1].data["code"] == "NO_PENDING_CONFIRMATION"
    assert await rig.drafts.list_for(ALICE) == []
    assert (await agent.pending_confirmation(ALICE, stop["session_id"]))["id"] == stop["id"]


async def test_resuming_without_a_pending_confirmation_is_an_error(rig: Rig):
    agent = rig.build(llm_for())
    events = await collect(agent.resume(ALICE, "nothing-here", "x", "approve"))
    assert events[-1].kind == "error" and events[-1].data["code"] == "NO_PENDING_CONFIRMATION"


async def test_a_new_message_drops_the_open_confirmation(rig: Rig):
    llm = llm_for({"tools": [REFUND]}, {"text": "Hello again."})
    llm.route = "combined"
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "refund order 1234"))
    llm.route = "chitchat"
    events = await ask(agent, "never mind, hi")
    assert events[-1].kind == "done"
    assert await agent.pending_confirmation(ALICE, stop["session_id"]) is None
    late = await answer_with(agent, stop, "approve")
    assert late[-1].kind == "error"
    assert await rig.drafts.list_for(ALICE) == []


# --- safety ---------------------------------------------------------------------------------------


async def test_someone_elses_confirmation_cannot_be_answered(rig: Rig):
    agent = rig.build(llm_for({"tools": [REFUND]}))
    stop = waiting(await ask(agent, "refund order 1234"))
    events = await collect(agent.resume(BOB, stop["session_id"], stop["id"], "approve"))
    assert events[-1].kind == "error"
    assert await rig.drafts.list_for(ALICE) == [] and await rig.drafts.list_for(BOB) == []


async def test_a_refund_for_someone_elses_order_is_never_offered(rig: Rig):
    foreign = ("propose_draft", REFUND[1] | {"order_id": "2001"})
    llm = llm_for({"tools": [foreign]}, {"text": "I could not find that order."})
    agent = rig.build(llm)
    events = await ask(agent, "refund order 2001")
    assert events[-1].kind == "done" and done(events)["drafts"] == []


async def test_asking_twice_does_not_make_two_requests(rig: Rig):
    llm = llm_for(
        {"tools": [REFUND]},
        {"text": "Submitted."},
        {"tools": [REFUND]},
        {"text": "You already have one open."},
    )
    agent = rig.build(llm)
    stop = waiting(await ask(agent, "refund order 1234", "a"))
    done(await answer_with(agent, stop, "approve"))
    again = await ask(agent, "refund order 1234 again", "a")
    assert again[-1].kind == "done" and done(again)["drafts"] == []  # never even offered
    assert len(await rig.drafts.list_for(ALICE)) == 1


async def test_the_model_cannot_propose_two_requests_in_one_step(rig: Rig):
    llm = llm_for({"tools": [REFUND, WARRANTY]}, {"text": "One at a time."})
    agent = rig.build(llm)
    events = await ask(agent, "refund and warranty for 1234")
    stop = waiting(events)
    assert stop["draft_type"] == "refund"


async def test_without_a_drafts_service_requests_are_unavailable(rig: Rig):
    llm = llm_for({"text": "Please contact the shop."})
    agent = rig.build(llm, with_drafts=False)
    done(await ask(agent, "refund order 1234"))
    offered = {t for bind in llm.agent_binds() for t in bind["tools"]}
    assert "propose_draft" not in offered and "cancel_draft" not in offered


# --- surviving a restart ---------------------------------------------------------------------------


async def test_a_confirmation_survives_a_restart(rig: Rig, tmp_path: Path):
    from support_agent.agent.checkpoint import open_checkpointer

    url = f"sqlite:///{(tmp_path / 'ck.db').as_posix()}"
    async with open_checkpointer(url) as saver:
        stop = waiting(
            await ask(rig.build(llm_for({"tools": [REFUND]}), checkpointer=saver), "refund 1234")
        )
    async with open_checkpointer(url) as saver:
        agent = rig.build(llm_for({"text": "Submitted."}), checkpointer=saver)
        assert (await agent.pending_confirmation(ALICE, stop["session_id"]))["id"] == stop["id"]
        done(await answer_with(agent, stop, "approve"))
    assert len(await rig.drafts.list_for(ALICE)) == 1


# --- the evaluation harness drives the whole request flow ------------------------------------------


def _sample(id_: str, text: str, **expected: Any):
    from support_agent.evals.dataset import Sample

    return Sample.model_validate(
        {
            "id": id_,
            "lang": "en",
            "type": "aftersales",
            "principal": {"user_id": "u_100"},
            "input": text,
            "expected": {"route": "combined", "tools": ["propose_draft"], **expected},
        }
    )


async def test_the_harness_confirms_scores_and_cleans_up_after_itself(rig: Rig):
    from support_agent.evals.runner import run_pipeline_eval, summarise_pipeline

    llm = llm_for(
        {"tools": [REFUND]},  # stop
        {"tools": [REFUND]},  # reject
        {"text": "Understood, nothing was sent."},
        {"tools": [REFUND]},  # approve
        {"text": "It is waiting for staff."},
    )
    samples = [
        _sample(
            "stop", "refund 1234", outcome="confirmation", draft_type="refund", draft_amount=350000
        ),
        _sample("no", "refund 1234", outcome="answered", draft_type="refund", decision="reject"),
        _sample("yes", "refund 1234", outcome="answered", draft_type="refund", decision="approve"),
    ]
    results = await run_pipeline_eval(rig.build(llm), samples, run_name="t", concurrency=1)
    report = summarise_pipeline(results)
    assert [r.scores.draft_ok for r in results] == [True, True, True], report.failures
    assert report.metrics["business"] == 1.0 and report.metrics["answer"] == 1.0
    assert len(await rig.drafts.list_for(ALICE)) == 1  # only the approved one
    assert results[2].answer.drafts and results[2].answer.interrupt["draft_type"] == "refund"
    assert results[0].answer.route == "combined" and results[0].answer.tool_calls == [
        "propose_draft"
    ]


async def test_a_wrong_amount_or_a_stray_draft_is_scored_as_wrong(rig: Rig):
    from support_agent.evals.runner import run_pipeline_eval

    agent = rig.build(llm_for({"tools": [REFUND]}, {"tools": [REFUND]}, {"text": "ok"}))
    wrong_amount = _sample(
        "a", "refund 1234", outcome="confirmation", draft_type="refund", draft_amount=1
    )
    unexpected = _sample("b", "refund 1234", outcome="confirmation", draft_type=None, tools=[])
    unexpected.expected.tools = ["propose_draft"]
    results = await run_pipeline_eval(agent, [wrong_amount, unexpected], run_name="t")
    assert [r.scores.draft_ok for r in results] == [False, False]
    assert not any(r.scores.answer_ok for r in results)


async def test_samples_that_create_drafts_run_last_and_in_file_order(rig: Rig):
    from support_agent.evals.runner import run_pipeline_eval

    order: list[str] = []
    agent = rig.build(llm_for())
    real = agent.answer

    async def spy(question: str, principal: Any, **kw: Any) -> Any:
        order.append(question)
        return await real(question, principal, **kw)

    agent.answer = spy  # type: ignore[method-assign]
    samples = [
        _sample("1", "approve-first", decision="approve", draft_type="refund"),
        _sample("2", "plain", outcome="answered"),
        _sample("3", "approve-second", decision="approve", draft_type="refund"),
    ]
    await run_pipeline_eval(agent, samples, run_name="t", concurrency=3)
    assert order == ["plain", "approve-first", "approve-second"]
