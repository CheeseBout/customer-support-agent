"""The after-sales evaluation dataset must agree with the demo shop, and be well formed.

Every confirmation the dataset expects (type, amount, priority flag) and every refusal it
expects is checked here against the real rules and the seeded data, with no model involved.
A dataset whose answers were wrong would measure nothing.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from support_agent.agent.drafting import DraftProposer, ProposalError, ProposeDraftArgs
from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig
from support_agent.drafts.repository import SqlDraftRepository
from support_agent.drafts.service import DraftService
from support_agent.evals.dataset import Sample, check_dataset, load_dataset
from support_agent.tools.client import DomainToolClient
from tests.conftest import DEMO

DATASET = DEMO / "evals" / "datasets" / "aftersales.jsonl"
SAMPLES = {s.id: s for s in load_dataset(DATASET)}


def req(kind: str, **kw: Any) -> ProposeDraftArgs:
    return ProposeDraftArgs(draft_type=kind, **kw)  # type: ignore[arg-type]


def refund(order: str, **kw: Any) -> ProposeDraftArgs:
    return req("refund", order_id=order, reason_code=kw.pop("reason", "defective"), **kw)


def ret(order: str, **kw: Any) -> ProposeDraftArgs:
    return req("return", order_id=order, reason_code=kw.pop("reason", "changed_mind"), **kw)


def warranty(order: str, sku: str, issue: str = "it does not work properly") -> ProposeDraftArgs:
    return req("warranty", order_id=order, sku=sku, issue_description=issue)


def order(items: list[dict[str, Any]], address: str, payment: str) -> ProposeDraftArgs:
    return req("order", items=items, shipping_address=address, payment_method=payment)


ADDR_A = "12 Lê Lợi, Quận 1, TP. Hồ Chí Minh"
ADDR_B = "45 Trần Hưng Đạo, Hà Nội"
ADDR_C = "9 Nguyễn Huệ, Đà Nẵng"
EAR = [{"sku": "EAR-BT20", "qty": 1}]
PHONE = [{"sku": "PHN-X100", "qty": 1}]
CASE = [{"sku": "CASE-X100", "qty": 1}]

# What the customer asks for, in each sample that should end in a confirmation. The oracle is
# the real proposer on the seeded shop; the dataset must say what it says.
PROPOSED: dict[str, tuple[str, ProposeDraftArgs]] = {
    "aft-en-001": ("u_100", refund("1234")),
    "aft-vi-001": ("u_100", refund("1234")),
    "aft-en-002": ("u_100", ret("1234")),
    "aft-vi-002": ("u_100", ret("1234")),
    "aft-en-003": ("u_101", refund("2001", items=PHONE, reason="defective")),
    "aft-vi-003": ("u_101", refund("2001", reason="defective")),
    "aft-en-004": ("u_101", ret("2001", items=PHONE, reason="not_as_described")),
    "aft-vi-004": ("u_101", ret("2001", items=PHONE, reason="wrong_item")),
    "aft-en-005": ("u_100", refund("1234")),
    "aft-vi-005": ("u_100", refund("1234")),
    "aft-en-006": ("u_100", warranty("1235", "CHG-65W")),
    "aft-vi-006": ("u_102", warranty("3001", "APL-AP20")),
    "aft-vi-007": ("u_101", warranty("2001", "PHN-X100")),
    "aft-en-007": ("u_102", warranty("3001", "APL-AP20", "the fan makes a grinding noise")),
    "aft-en-008": ("u_100", order([{"sku": "MOU-M10", "qty": 2}], ADDR_A, "momo")),
    "aft-vi-022": (
        "u_100",
        order([{"sku": "MOU-M10", "qty": 1}], ADDR_A, "cod"),
    ),  # "my old address"
    "aft-vi-008": ("u_101", order([{"sku": "LAP-AIR13", "qty": 1}], ADDR_B, "bank_transfer")),
    "aft-en-009": (
        "u_100",
        order([{"sku": "SPK-BM5", "qty": 1}, {"sku": "CAB-USBC2", "qty": 1}], ADDR_A, "cod"),
    ),
    "aft-vi-009": ("u_102", order([{"sku": "MOU-M10", "qty": 3}], ADDR_C, "zalopay")),
    "aft-vi-027": ("u_100", order([{"sku": "CAB-USBC2", "qty": 1}], ADDR_A, "momo")),
    "aft-en-010": ("u_101", refund("2001", items=PHONE, reason="defective")),
    "aft-vi-010": ("u_100", warranty("1234", "EAR-BT20", "pin tụt nhanh")),
    "aft-en-011": ("u_100", order([{"sku": "KEY-K75", "qty": 1}], ADDR_A, "bank_transfer")),
    "aft-en-012": ("u_101", refund("2001", items=PHONE, reason="defective")),
    "aft-vi-011": ("u_100", warranty("1235", "CHG-65W", "củ sạc bị nóng")),
    "aft-en-013": ("u_100", refund("1234")),
    "aft-vi-012": ("u_101", ret("2001", items=CASE)),
    "aft-en-014": ("u_100", warranty("1234", "EAR-BT20", "the battery drains in an hour")),
    "aft-vi-013": ("u_102", warranty("3001", "APL-AP20", "chạy rất ồn")),
    "aft-en-015": (
        "u_100",
        order(
            [{"sku": "CASE-X100", "qty": 1}, {"sku": "MOU-M10", "qty": 1}], ADDR_A, "bank_transfer"
        ),
    ),
    "aft-vi-014": ("u_101", order([{"sku": "KEY-K75", "qty": 2}], ADDR_B, "cod")),
    "aft-en-030": ("u_100", refund("1234")),
    "aft-vi-026": ("u_100", refund("1234", reason="other")),
    "aft-en-031": ("u_100", refund("1234")),
}

# What the customer asks for in each sample where the shop must say no.
REFUSED: dict[str, tuple[str, ProposeDraftArgs, str]] = {
    "aft-en-018": ("u_100", refund("1235"), "NOT_ELIGIBLE"),  # 20 days since delivery
    "aft-vi-017": ("u_100", refund("1236", reason="changed_mind"), "NOT_ELIGIBLE"),  # in transit
    "aft-en-019": ("u_100", ret("1239"), "NOT_ELIGIBLE"),  # gift card
    "aft-vi-018": ("u_100", ret("1240"), "NOT_ELIGIBLE"),  # a request is already pending
    "aft-en-020": ("u_100", warranty("1236", "LAP-PRO14"), "NOT_ELIGIBLE"),  # not delivered
    "aft-vi-019": ("u_100", refund("1238", reason="changed_mind"), "NOT_ELIGIBLE"),  # cancelled
    "aft-en-021": ("u_100", refund("1237", reason="changed_mind"), "NOT_ELIGIBLE"),  # processing
    "aft-vi-020": ("u_102", ret("3001"), "NOT_ELIGIBLE"),  # 10 days since delivery
    "aft-en-022": ("u_100", warranty("1234", "LAP-PRO14"), "NOT_ELIGIBLE"),  # not on the order
    "aft-en-025": (
        "u_101",
        order([{"sku": "LAP-AIR13", "qty": 1}], ADDR_B, "cod"),
        "NOT_ELIGIBLE",
    ),
    "aft-vi-023": ("u_100", order([{"sku": "CHG-65W", "qty": 1}], ADDR_A, "momo"), "OUT_OF_STOCK"),
    "aft-en-026": (
        "u_100",
        order([{"sku": "LAP-PRO14", "qty": 5}], ADDR_A, "card"),
        "OUT_OF_STOCK",
    ),
    "aft-vi-024": (
        "u_102",
        order([{"sku": "MOU-M10", "qty": 25}], ADDR_C, "bank_transfer"),
        "LIMIT_EXCEEDED",
    ),
    "aft-en-027": (
        "u_100",
        order([{"sku": "KEY-K75", "qty": 1}], ADDR_A, "bitcoin"),
        "NOT_ELIGIBLE",
    ),
    "aft-en-028": ("u_100", refund("2001"), "NOT_FOUND"),  # someone else's order
    "aft-vi-025": ("u_100", warranty("2002", "APL-AF30"), "NOT_FOUND"),
    "aft-en-029": ("u_100", ret("9999"), "NOT_FOUND"),
}


@pytest_asyncio.fixture
async def proposer(
    tool_client: DomainToolClient, app_config: AppConfig, tmp_path: Path
) -> AsyncIterator[DraftProposer]:
    repo = SqlDraftRepository.from_url(f"sqlite+aiosqlite:///{(tmp_path / 'd.db').as_posix()}")
    await repo.create_schema()
    drafts = DraftService(repo, app_config.business_rules)
    yield DraftProposer(tool_client, drafts, app_config.business_rules)
    await repo.close()


# --- the file itself -----------------------------------------------------------------------------


def test_the_dataset_is_well_formed_and_balanced():
    samples = list(SAMPLES.values())
    assert len(samples) >= 60 and check_dataset(samples) == []
    vi = sum(1 for s in samples if s.lang == "vi")
    assert 0.4 <= vi / len(samples) <= 0.6
    assert all(s.type == "aftersales" for s in samples)


def test_every_kind_of_request_and_every_failure_mode_is_covered():
    tags = {t for s in SAMPLES.values() for t in s.tags}
    kinds = {s.expected.draft_type for s in SAMPLES.values()} - {None}
    assert kinds == {"refund", "return", "warranty", "order"}
    decisions = {s.expected.decision for s in SAMPLES.values()}
    assert decisions == {None, "approve", "reject", "edit"}
    assert {
        "ineligible", "missing_info", "foreign_order", "injection", "indirect_injection",
        "price_tampering", "self_approval", "stock_privacy", "cod_limit", "multi_turn",
        "priority", "after_approvals", "guidance", "duplicate", "window_expired",
    } <= tags  # fmt: skip


def test_the_ci_subset_has_every_flow_and_both_languages():
    ci = [s for s in SAMPLES.values() if "ci" in s.subset]
    assert 12 <= len(ci) <= 24
    assert {s.lang for s in ci} == {"vi", "en"}
    assert {s.expected.decision for s in ci} == {None, "approve", "reject", "edit"}
    # a sample that reads drafts back needs the approvals it reads to be in the subset too
    assert any("after_approvals" in s.tags for s in ci)
    assert any(s.expected.decision == "approve" and s.principal.user_id == "u_100" for s in ci)


def test_samples_with_no_expected_request_never_expect_a_decision():
    for s in SAMPLES.values():
        if s.expected.draft_type is None:
            assert s.expected.decision is None and not s.expected.edits, s.id
        if s.expected.edits:
            assert s.expected.decision == "edit", s.id


def test_every_sample_that_should_stop_or_refuse_has_an_oracle_entry():
    assert {i for i, x in SAMPLES.items() if x.expected.draft_type} == set(PROPOSED)
    assert all(SAMPLES[i].expected.draft_type is None for i in REFUSED)
    refusal_tags = {"ineligible", "cod_limit", "out_of_stock", "stock_privacy", "quantity_limit"}
    refusal_tags |= {"payment_method", "foreign_order", "missing_order"}
    expected = {
        i
        for i, x in SAMPLES.items()
        if refusal_tags & set(x.tags)
        and x.expected.draft_type is None
        and "warranty_check" not in x.tags
    }
    assert expected == set(REFUSED)


def test_approvals_never_target_the_same_request_twice():
    seen: set[tuple[str, str, str | None]] = set()
    for s in SAMPLES.values():
        if s.expected.decision == "approve" and s.expected.draft_type in ("refund", "return"):
            key = (s.principal.user_id, s.expected.draft_type, s.id)
            assert key not in seen
            seen.add(key)
    approving = [
        (s.principal.user_id, PROPOSED[s.id][1].order_id, PROPOSED[s.id][1].sku)
        for s in SAMPLES.values()
        if s.expected.decision == "approve" and s.expected.draft_type in ("refund", "return")
    ]
    assert len(approving) == len(set(approving)), "two approvals would block each other"


# --- the dataset against the shop ----------------------------------------------------------------


def _principal(user: str) -> Principal:
    return Principal(user_id=user)


@pytest.mark.parametrize("sample_id", sorted(PROPOSED))
async def test_expected_confirmations_match_what_the_shop_computes(
    sample_id: str, proposer: DraftProposer
):
    sample: Sample = SAMPLES[sample_id]
    user, args = PROPOSED[sample_id]
    exp = sample.expected
    assert sample.principal.user_id == user
    proposal = await proposer.propose(args, _principal(user))
    if exp.decision == "edit":
        proposal = await proposer.edited(proposal, exp.edits, _principal(user))
    assert proposal.draft_type == exp.draft_type
    shown = proposal.summary.get("amount", proposal.summary.get("total"))
    if exp.draft_amount is not None:
        assert shown == exp.draft_amount
    if exp.priority_review is not None:
        assert proposal.priority_review is exp.priority_review
    text = str(proposal.summary).casefold()
    assert all(part.casefold() in text for part in exp.summary_contains)


@pytest.mark.parametrize("sample_id", sorted(REFUSED))
async def test_expected_refusals_are_refused_by_the_shop(sample_id: str, proposer: DraftProposer):
    user, args, code = REFUSED[sample_id]
    assert SAMPLES[sample_id].principal.user_id == user
    with pytest.raises(ProposalError) as raised:
        await proposer.propose(args, _principal(user))
    assert raised.value.code == code
