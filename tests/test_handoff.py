"""Passing a customer to a person: a request they confirm, then a pending item for staff."""

from __future__ import annotations

from typing import Any

import pytest

from support_agent.agent.drafting import DraftProposer, ProposalError, ProposeDraftArgs
from support_agent.agent.skills import load_skill, skill_catalog
from support_agent.agent.tools import draft_view
from support_agent.core.capabilities import Offer
from support_agent.drafts.models import PAYLOAD_MODELS
from tests.conftest import ALICE, BOB, STAFF
from tests.test_agent import fast_config as fast_config  # noqa: F401
from tests.test_agent import kinds
from tests.test_agent import retriever as retriever  # noqa: F401
from tests.test_agent_actions import Rig, answer_with, ask, llm_for, rig, waiting  # noqa: F401

HANDOFF = (
    "propose_draft",
    {
        "draft_type": "handoff",
        "order_id": "1236",
        "reason_text": "The parcel says delivered but I never got it",
    },
)


# --- the proposal ------------------------------------------------------------------------------------------


async def test_the_customer_is_asked_to_confirm_before_anything_is_created(rig: Rig):  # noqa: F811
    agent = rig.build(llm_for({"tools": [HANDOFF]}))
    stop = waiting(await ask(agent, "I want to talk to someone about order 1236"))
    assert stop["draft_type"] == "handoff"
    assert stop["summary"] == {
        "reason": "The parcel says delivered but I never got it",
        "order_id": "1236",
    }
    assert stop["editable_fields"] == ["contact", "reason_text"]
    assert await rig.drafts.list_for(ALICE) == []  # asking is not submitting


async def test_confirming_puts_a_pending_item_in_the_staff_queue(rig: Rig):  # noqa: F811
    agent = rig.build(llm_for({"tools": [HANDOFF]}, {"text": "Staff will follow up."}))
    stop = waiting(await ask(agent, "I want to talk to someone about order 1236"))
    events = await answer_with(agent, stop, "approve")
    assert "done" in kinds(events)

    (draft,) = await rig.drafts.list_for(ALICE)
    assert draft.type == "handoff" and draft.status == "pending"
    assert draft.payload == {
        "reason": "The parcel says delivered but I never got it",
        "order_id": "1236",
        "contact": "",
    }
    queue = await rig.drafts.queue(STAFF)
    assert [d.id for d in queue] == [draft.id]
    assert PAYLOAD_MODELS["handoff"].model_validate(draft.payload)


async def test_the_customer_can_add_a_contact_while_confirming(rig: Rig):  # noqa: F811
    agent = rig.build(llm_for({"tools": [HANDOFF]}, {"text": "Done."}))
    stop = waiting(await ask(agent, "I want to talk to someone"))
    again = waiting(await answer_with(agent, stop, "edit", contact="  an@example.test "))
    assert again["summary"]["contact"] == "an@example.test"  # shown again, tidied, before saving
    await answer_with(agent, again, "approve")
    (draft,) = await rig.drafts.list_for(ALICE)
    assert draft.payload["contact"] == "an@example.test"


async def test_declining_creates_nothing(rig: Rig):  # noqa: F811
    agent = rig.build(llm_for({"tools": [HANDOFF]}, {"text": "No problem."}))
    stop = waiting(await ask(agent, "I want to talk to someone"))
    await answer_with(agent, stop, "reject")
    assert await rig.drafts.list_for(ALICE) == []


async def test_staff_can_work_the_item_like_any_other_request(rig: Rig):  # noqa: F811
    agent = rig.build(llm_for({"tools": [HANDOFF]}, {"text": "Done."}))
    stop = waiting(await ask(agent, "I want to talk to someone about order 1236"))
    await answer_with(agent, stop, "approve")
    (draft,) = await rig.drafts.list_for(ALICE)
    handled = await rig.drafts.review(STAFF, draft.id, "approve", "Called the customer back")
    assert handled.status == "approved" and handled.review_note == "Called the customer back"
    assert draft_view(handled)["staff_note"] == "Called the customer back"


# --- what is refused --------------------------------------------------------------------------------------------


async def test_an_order_that_is_not_the_customers_cannot_be_attached(rig: Rig):  # noqa: F811
    stolen = ("propose_draft", {**HANDOFF[1], "order_id": "2001"})  # Bob's order
    agent = rig.build(llm_for({"tools": [stolen]}, {"text": "I could not find that order."}))
    events = await ask(agent, "talk to someone about 2001", who=ALICE)
    assert not any(e.kind == "interrupt" for e in events)
    assert await rig.drafts.list_for(ALICE) == []
    assert await rig.drafts.list_for(BOB) == []


@pytest.mark.parametrize("reason", [None, "", "  ", "ok"])
async def test_a_reason_in_the_customers_words_is_required(
    rig: Rig,  # noqa: F811
    tool_client,
    reason: str | None,
):
    args: dict[str, Any] = {"draft_type": "handoff"}
    if reason is not None:
        args["reason_text"] = reason
    direct = DraftProposer(tool_client, rig.drafts, rig.drafts.rules)
    with pytest.raises(ProposalError) as exc:
        await direct.propose(ProposeDraftArgs.model_validate(args), ALICE)
    assert exc.value.code == "INVALID_ARGUMENT" and "reason_text" in exc.value.message


async def test_a_handoff_needs_no_order(rig: Rig, tool_client):  # noqa: F811
    direct = DraftProposer(tool_client, rig.drafts, rig.drafts.rules)
    proposal = await direct.propose(
        ProposeDraftArgs(draft_type="handoff", reason_text="I need to change my phone number"),
        ALICE,
    )
    assert proposal.payload["order_id"] is None and "order_id" not in proposal.summary


# --- what the model is told -------------------------------------------------------------------------------------------


def test_the_skill_is_offered_only_when_handoff_is_enabled():
    assert "talk-to-a-person" in {n for n, _ in skill_catalog()}
    off = Offer(request_types=("order", "refund", "return", "warranty"))
    assert "talk-to-a-person" not in {n for n, _ in skill_catalog(off)}


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_the_skill_never_invents_a_reason_or_a_contact(lang: str):
    skill = load_skill("talk-to-a-person", lang)
    assert skill is not None and skill.requires == ("request:handoff",)
    assert "propose_draft" in skill.body and "handoff" in skill.body
