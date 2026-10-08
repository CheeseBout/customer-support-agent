from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from mongomock_motor import AsyncMongoMockClient

from support_agent.core.principal import Principal
from support_agent.core.settings import AppConfig
from support_agent.drafts.models import Draft, idempotency_key
from support_agent.drafts.repository import (
    DuplicateKey,
    MongoDraftRepository,
    SqlDraftRepository,
    create_repository,
)
from support_agent.drafts.service import DraftError, DraftService
from tests.conftest import ALICE, BOB, STAFF

T0 = datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC)


def refund_payload(**over: Any) -> dict[str, Any]:
    base = {
        "order_id": "1234",
        "items": [{"sku": "EAR-BT20", "qty": 1}],
        "reason_code": "defective",
        "reason_text": "left bud is dead",
        "refundable_amount": 350_000,
        "eligibility_snapshot": {"eligible": True, "days_remaining": 3},
    }
    return base | over


WARRANTY = {"order_id": "1234", "sku": "EAR-BT20", "issue_description": "battery drains fast"}
ORDER_DRAFT = {
    "items": [{"sku": "PHN-X100", "name": "Phone", "qty": 1, "unit_price": 7_990_000}],
    "shipping_address": "12 Le Loi, District 1",
    "payment_method": "cod",
    "total": 7_990_000,
    "currency": "VND",
}


class Clock:
    """A clock the tests can move."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw: float) -> None:
        self.now += timedelta(**kw)


@pytest_asyncio.fixture(params=["sql", "mongo"])
async def repo(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Any]:
    if request.param == "sql":
        r: Any = SqlDraftRepository.from_url(
            f"sqlite+aiosqlite:///{(tmp_path / 'd.db').as_posix()}"
        )
    else:
        r = MongoDraftRepository(AsyncMongoMockClient(tz_aware=True)["shop"])
    await r.create_schema()
    yield r
    await r.close()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(repo: Any, app_config: AppConfig, clock: Clock) -> DraftService:
    counter = iter(range(1, 10_000))
    return DraftService(
        repo, app_config.business_rules, clock=clock, new_id=lambda: f"d{next(counter):03d}"
    )


# --- creating -------------------------------------------------------------------------------------


async def test_create_makes_a_pending_draft_owned_by_the_caller(service: DraftService):
    draft, created = await service.create(ALICE, "refund", refund_payload(), "s1")
    assert created and draft.id == "d001" and draft.status == "pending"
    assert draft.customer_id == "u_100" and draft.session_id == "s1" and draft.type == "refund"
    assert draft.payload["order_id"] == "1234" and draft.created_at == T0 == draft.updated_at
    assert draft.reviewed_by is None and draft.reviewed_at is None and not draft.priority_review


@pytest.mark.parametrize(
    "amount,flagged",
    [(500_000, False), (500_001, True), (100, False), (18_900_000, True)],
)
async def test_large_refunds_are_flagged_for_priority_review(service, amount, flagged):
    draft, _ = await service.create(ALICE, "refund", refund_payload(refundable_amount=amount), "s")
    assert draft.priority_review is flagged and draft.status == "pending"  # never auto-approved


async def test_only_refunds_get_the_priority_flag(service: DraftService):
    ret, _ = await service.create(ALICE, "return", refund_payload(refundable_amount=9_000_000), "s")
    assert not ret.priority_review


async def test_every_draft_type_round_trips(service: DraftService):
    for kind, payload in (
        ("order", ORDER_DRAFT),
        ("warranty", WARRANTY),
        ("return", refund_payload()),
    ):
        draft, _ = await service.create(ALICE, kind, payload, f"s-{kind}")
        assert (await service.get_for(ALICE, draft.id)).payload == draft.payload


async def test_identical_requests_return_the_same_draft(service: DraftService):
    first, created1 = await service.create(ALICE, "refund", refund_payload(), "s1")
    again, created2 = await service.create(ALICE, "refund", refund_payload(), "s1")
    assert created1 and not created2 and again.id == first.id
    assert len(await service.list_for(ALICE)) == 1


async def test_the_eligibility_snapshot_does_not_change_the_idempotency_key(service: DraftService):
    first, _ = await service.create(ALICE, "refund", refund_payload(), "s1")
    later = refund_payload(eligibility_snapshot={"eligible": True, "days_remaining": 2})  # a day on
    again, created = await service.create(ALICE, "refund", later, "s1")
    assert not created and again.id == first.id


async def test_a_different_request_or_session_makes_a_new_draft(service: DraftService):
    a, _ = await service.create(ALICE, "refund", refund_payload(), "s1")
    b, _ = await service.create(ALICE, "refund", refund_payload(reason_text="other"), "s1")
    c, _ = await service.create(ALICE, "refund", refund_payload(), "s2")
    assert len({a.id, b.id, c.id}) == 3


async def test_two_customers_with_the_same_session_id_never_share_a_draft(service: DraftService):
    mine, _ = await service.create(ALICE, "warranty", WARRANTY, "same-session")
    theirs, created = await service.create(BOB, "warranty", WARRANTY, "same-session")
    assert created and theirs.id != mine.id and theirs.customer_id == "u_101"


async def test_concurrent_identical_requests_create_one_draft(service: DraftService):
    results = await asyncio.gather(
        *(service.create(ALICE, "refund", refund_payload(), "race") for _ in range(6))
    )
    assert {d.id for d, _ in results}.__len__() == 1
    assert sum(1 for _, created in results if created) == 1
    assert len(await service.list_for(ALICE)) == 1


@pytest.mark.parametrize(
    "kind,payload",
    [
        ("refund", refund_payload(items=[])),
        ("refund", refund_payload(reason_code="because")),
        ("refund", refund_payload(surprise=1)),  # unknown fields are rejected, not stored
        ("refund", {k: v for k, v in refund_payload().items() if k != "order_id"}),
        ("warranty", WARRANTY | {"issue_description": "x"}),
        ("order", ORDER_DRAFT | {"items": []}),
        ("order", ORDER_DRAFT | {"shipping_address": ""}),
        ("gift", {}),
    ],
)
async def test_invalid_requests_are_refused_with_the_field_names(service, kind, payload):
    with pytest.raises(DraftError) as exc:
        await service.create(ALICE, kind, payload, "s")
    assert exc.value.code == "INVALID_ARGUMENT"


async def test_drafts_keep_their_session_for_traceability(service: DraftService):
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "session-xyz")
    assert (await service.get_for(ALICE, draft.id)).session_id == "session-xyz"


# --- reading ---------------------------------------------------------------------------------------


async def test_a_customer_lists_only_their_own_drafts_newest_first(
    service: DraftService, clock: Clock
):
    first, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    clock.advance(minutes=5)
    second, _ = await service.create(ALICE, "refund", refund_payload(), "b")
    await service.create(BOB, "warranty", WARRANTY | {"order_id": "2001"}, "c")
    assert [d.id for d in await service.list_for(ALICE)] == [second.id, first.id]
    assert [d.customer_id for d in await service.list_for(BOB)] == ["u_101"]


async def test_listing_filters_by_status_and_caps_the_page(service: DraftService):
    d1, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    await service.create(ALICE, "refund", refund_payload(), "b")
    await service.cancel(ALICE, d1.id)
    assert [d.status for d in await service.list_for(ALICE, statuses=["cancelled"])] == [
        "cancelled"
    ]
    assert len(await service.list_for(ALICE, statuses=["pending"])) == 1
    assert (
        len(await service.list_for(ALICE, limit=0)) >= 1
        and len(await service.list_for(ALICE, limit=9999)) == 2
    )


async def test_someone_elses_draft_looks_exactly_like_a_missing_one(service: DraftService):
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    with pytest.raises(DraftError) as foreign:
        await service.get_for(BOB, draft.id)
    with pytest.raises(DraftError) as missing:
        await service.get_for(BOB, "no-such-id")
    assert foreign.value.code == missing.value.code == "NOT_FOUND"
    assert foreign.value.message.replace(draft.id, "X") == missing.value.message.replace(
        "no-such-id", "X"
    )
    assert (await service.get_for(STAFF, draft.id)).id == draft.id  # staff can see any


# --- cancelling ---------------------------------------------------------------------------------------


async def test_the_owner_can_cancel_a_pending_draft(service: DraftService, clock: Clock):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    clock.advance(minutes=1)
    cancelled = await service.cancel(ALICE, draft.id)
    assert cancelled.status == "cancelled" and cancelled.updated_at == clock.now
    assert cancelled.reviewed_by is None  # a withdrawal is not a staff decision


async def test_nobody_else_can_cancel_and_cancelling_twice_conflicts(service: DraftService):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    with pytest.raises(DraftError) as foreign:
        await service.cancel(BOB, draft.id)
    assert foreign.value.code == "NOT_FOUND"
    await service.cancel(ALICE, draft.id)
    with pytest.raises(DraftError) as again:
        await service.cancel(ALICE, draft.id)
    assert again.value.code == "CONFLICT"


async def test_an_approved_draft_can_no_longer_be_cancelled(service: DraftService):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    await service.review(STAFF, draft.id, "approve")
    with pytest.raises(DraftError) as exc:
        await service.cancel(ALICE, draft.id)
    assert exc.value.code == "CONFLICT"


# --- staff review ----------------------------------------------------------------------------------------


async def test_staff_approve_and_the_decision_is_recorded(service: DraftService, clock: Clock):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    clock.advance(hours=2)
    approved = await service.review(STAFF, draft.id, "approve", "verified the photos")
    assert approved.status == "approved" and approved.reviewed_by == "s_1"
    assert approved.reviewed_at == clock.now and approved.review_note == "verified the photos"


async def test_rejecting_requires_a_note_the_customer_can_be_told(service: DraftService):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    for note in (None, "", "   "):
        with pytest.raises(DraftError) as exc:
            await service.review(STAFF, draft.id, "reject", note)
        assert exc.value.code == "INVALID_ARGUMENT"
    rejected = await service.review(STAFF, draft.id, "reject", "item shows physical damage")
    assert rejected.status == "rejected" and rejected.review_note == "item shows physical damage"


async def test_customers_cannot_review_even_their_own_drafts(service: DraftService):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    with pytest.raises(DraftError) as exc:
        await service.review(ALICE, draft.id, "approve")
    assert exc.value.code == "FORBIDDEN"
    assert (await service.get_for(ALICE, draft.id)).status == "pending"


@pytest.mark.parametrize("first", ["approve", "reject"])
async def test_a_decided_draft_cannot_be_decided_again(service: DraftService, first: str):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    await service.review(STAFF, draft.id, first, "because")
    for second in ("approve", "reject"):
        with pytest.raises(DraftError) as exc:
            await service.review(STAFF, draft.id, second, "again")
        assert exc.value.code == "CONFLICT"


async def test_two_staff_deciding_at_once_produce_exactly_one_decision(service: DraftService):
    draft, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    outcomes = await asyncio.gather(
        service.review(STAFF, draft.id, "approve"),
        service.review(Principal(user_id="s_2", role="staff"), draft.id, "reject", "no"),
        return_exceptions=True,
    )
    winners = [o for o in outcomes if isinstance(o, Draft)]
    losers = [o for o in outcomes if isinstance(o, DraftError)]
    assert len(winners) == 1 and len(losers) == 1 and losers[0].code == "CONFLICT"
    assert (await service.get_for(STAFF, draft.id)).status == winners[0].status


async def test_the_review_queue_is_staff_only_oldest_first_and_filterable(service, clock):
    first, _ = await service.create(ALICE, "refund", refund_payload(refundable_amount=900_000), "a")
    clock.advance(minutes=1)
    second, _ = await service.create(BOB, "warranty", WARRANTY | {"order_id": "2001"}, "b")
    clock.advance(minutes=1)
    third, _ = await service.create(ALICE, "return", refund_payload(order_id="1235"), "c")
    await service.review(STAFF, third.id, "approve")

    with pytest.raises(DraftError) as exc:
        await service.queue(ALICE)
    assert exc.value.code == "FORBIDDEN"
    queue = await service.queue(STAFF)
    assert [d.id for d in queue] == [first.id, second.id] and queue[0].priority_review
    assert [d.id for d in await service.queue(STAFF, draft_type="warranty")] == [second.id]
    assert [d.id for d in await service.queue(STAFF, statuses=["approved"])] == [third.id]
    assert len(await service.queue(STAFF, limit=1, offset=1)) == 1


# --- feeding the eligibility rule -----------------------------------------------------------------------------


async def test_active_returns_and_refunds_block_a_second_request_for_the_same_item(
    service: DraftService,
):
    await service.create(ALICE, "refund", refund_payload(), "a")
    assert await service.existing_requests("u_100", "1234") == [
        {"status": "pending", "sku": "EAR-BT20"}
    ]


async def test_cancelled_rejected_other_orders_and_warranty_do_not_block(service: DraftService):
    cancelled, _ = await service.create(ALICE, "refund", refund_payload(), "a")
    await service.cancel(ALICE, cancelled.id)
    rejected, _ = await service.create(ALICE, "return", refund_payload(reason_text="x"), "b")
    await service.review(STAFF, rejected.id, "reject", "no")
    await service.create(ALICE, "warranty", WARRANTY, "c")
    await service.create(ALICE, "refund", refund_payload(order_id="1235"), "d")
    await service.create(BOB, "refund", refund_payload(), "e")  # same order id, another customer
    assert await service.existing_requests("u_100", "1234") == []


async def test_an_approved_request_still_blocks(service: DraftService):
    draft, _ = await service.create(ALICE, "return", refund_payload(), "a")
    await service.review(STAFF, draft.id, "approve")
    assert [r["status"] for r in await service.existing_requests("u_100", "1234")] == ["approved"]


# --- the repositories themselves -------------------------------------------------------------------------------


async def test_timestamps_come_back_as_utc_aware(service: DraftService, repo: Any):
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    stored = await repo.get(draft.id)
    assert stored.created_at.tzinfo is not None and stored.created_at == T0


async def test_a_duplicate_key_is_reported_by_the_repository(service: DraftService, repo: Any):
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    clone = draft.model_copy(update={"id": "other-id"})
    with pytest.raises(DuplicateKey):
        await repo.insert(clone)


async def test_transition_only_applies_from_the_expected_status(service: DraftService, repo: Any):
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "a")
    assert await repo.transition(draft.id, expected="approved", to="rejected", now=T0) is None
    assert (await repo.get(draft.id)).status == "pending"  # untouched
    moved = await repo.transition(draft.id, expected="pending", to="cancelled", now=T0)
    assert moved is not None and moved.status == "cancelled"


async def test_create_schema_can_run_twice(repo: Any):
    await repo.create_schema()
    await repo.create_schema()


async def test_unknown_ids_are_simply_absent(repo: Any):
    assert await repo.get("nope") is None and await repo.get_by_key("nope") is None


def test_idempotency_keys_depend_on_customer_session_type_and_content():
    base = idempotency_key("u1", "s", "refund", {"a": 1})
    assert base == idempotency_key("u1", "s", "refund", {"a": 1})
    assert len({base, idempotency_key("u2", "s", "refund", {"a": 1}), idempotency_key("u1", "t", "refund", {"a": 1}),
                idempotency_key("u1", "s", "return", {"a": 1}), idempotency_key("u1", "s", "refund", {"a": 2})}) == 5  # fmt: skip
    assert idempotency_key("u1", "s", "refund", {"b": 2, "a": 1}) == idempotency_key(
        "u1", "s", "refund", {"a": 1, "b": 2}
    )


@pytest.mark.parametrize(
    "url,cls,fragment",
    [
        ("sqlite:///x.db", SqlDraftRepository, "aiosqlite"),
        ("postgresql://u:p@h/db", SqlDraftRepository, "asyncpg"),
        ("mysql://u:p@h/db", SqlDraftRepository, "aiomysql"),
        ("mongodb://h/shop", MongoDraftRepository, ""),
    ],
)
def test_create_repository_picks_the_implementation_from_the_url(url, cls, fragment):
    made: Callable[[], Any] = lambda: create_repository(url)  # noqa: E731
    repo = made()
    assert isinstance(repo, cls)
    if fragment:
        assert fragment in str(repo.engine.url)


def test_create_repository_rejects_unknown_schemes():
    with pytest.raises(ValueError, match="Unsupported"):
        create_repository("redis://localhost")
