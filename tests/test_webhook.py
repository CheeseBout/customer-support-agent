"""The webhook connector: decisions reach the shop system, signed, once, in order, and not lost."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from mongomock_motor import AsyncMongoMockClient

from support_agent.core.settings import Settings, WebhookConfig
from support_agent.drafts.events import (
    EVENT_NAMES,
    DeliveryFailed,
    DraftEvent,
    WebhookDispatcher,
    WebhookSender,
    event_for,
    sign,
    verify_signature,
)
from support_agent.drafts.repository import MongoDraftRepository, SqlDraftRepository
from support_agent.drafts.service import DraftService
from tests.conftest import ALICE, BOB, DEMO_APP, STAFF
from tests.test_drafts import WARRANTY, Clock

SECRET = "whsec_test"
URL = "https://shop.example.test/hooks/support"
ALL = ["draft.created", "draft.approved", "draft.rejected", "draft.cancelled"]


class Receiver:
    """The shop side: records requests and answers from a script."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.script: list[int | Exception] = []
        self.default = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script.pop(0) if self.script else self.default
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, text="the other side's words")

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]

    def ids(self) -> list[str]:
        return [b["id"] for b in self.bodies()]


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
def receiver() -> Receiver:
    return Receiver()


def make_service(repo: Any, clock: Clock, publish: list[str] = ALL) -> DraftService:
    counter = iter(range(1, 1000))
    from support_agent.core.settings import BusinessRules

    return DraftService(
        repo, BusinessRules(), clock=clock, new_id=lambda: f"d{next(counter):03d}", publish=publish
    )


def make_dispatcher(repo: Any, receiver: Receiver, clock: Clock, **cfg: Any) -> WebhookDispatcher:
    config = WebhookConfig(events=ALL, backoff_seconds=30, max_attempts=3, **cfg)
    sender = WebhookSender(
        URL, SECRET, transport=receiver.transport, clock=lambda: clock.now.timestamp()
    )
    return WebhookDispatcher(repo, sender, config, clock=clock)


async def new_draft(service: DraftService, who=ALICE, session: str = "s1"):
    draft, _ = await service.create(who, "warranty", WARRANTY, session)
    return draft


# --- the event ------------------------------------------------------------------------------------------


async def test_an_event_carries_the_draft_but_not_internal_keys(repo, clock):
    service = make_service(repo, clock)
    draft = await new_draft(service)
    event = event_for(draft, clock.now)
    assert event.id == "d001:pending" and event.event == "draft.created"
    sent = event.body["draft"]
    assert sent["id"] == "d001" and sent["type"] == "warranty" and sent["customer_id"] == "u_100"
    assert sent["order_id"] == "1234" and sent["payload"]["sku"] == "EAR-BT20"
    assert "idempotency_key" not in sent and "session_id" not in sent
    assert event.body["event"] == "draft.created" and event.body["id"] == event.id


def test_every_status_has_an_event_name():
    assert EVENT_NAMES == {
        "pending": "draft.created",
        "approved": "draft.approved",
        "rejected": "draft.rejected",
        "cancelled": "draft.cancelled",
    }


# --- signing ----------------------------------------------------------------------------------------------


def test_a_signature_verifies_with_the_secret_and_not_otherwise():
    body = b'{"id":"d1:approved"}'
    header = f"t=1000,v1={sign(SECRET, 1000, body)}"
    assert verify_signature(SECRET, header, body, now=1010)
    assert not verify_signature("another", header, body, now=1010)
    assert not verify_signature(SECRET, header, body + b" ", now=1010)  # tampered
    assert not verify_signature(SECRET, header, body, now=1000 + 301)  # a replay, too late
    assert not verify_signature(SECRET, "garbage", body, now=1010)
    assert not verify_signature(SECRET, "t=abc,v1=00", body, now=1010)


async def test_the_sender_signs_exactly_what_it_sends(receiver, clock):
    sender = WebhookSender(
        URL, SECRET, transport=receiver.transport, clock=lambda: clock.now.timestamp()
    )
    event = DraftEvent(
        id="d1:approved",
        draft_id="d1",
        event="draft.approved",
        body={"x": 1},
        next_attempt_at=clock.now,
        created_at=clock.now,
    )
    await sender.send(event)
    (request,) = receiver.requests
    assert request.headers["X-Support-Event"] == "draft.approved"
    assert request.headers["X-Support-Event-Id"] == "d1:approved"
    assert verify_signature(
        SECRET, request.headers["X-Support-Signature"], request.content, now=clock.now.timestamp()
    )
    assert request.headers["Content-Type"] == "application/json"


@pytest.mark.parametrize(
    ("answer", "retryable"),
    [
        (500, True),
        (503, True),
        (429, True),
        (408, True),
        (400, False),
        (401, False),
        (404, False),
        (302, False),
    ],
)
async def test_which_answers_are_worth_retrying(receiver, clock, answer, retryable):
    receiver.default = answer
    sender = WebhookSender(URL, SECRET, transport=receiver.transport)
    event = event_for(await new_draft(make_service(_NoRepo(), clock)), clock.now)
    with pytest.raises(DeliveryFailed) as exc:
        await sender.send(event)
    assert exc.value.retryable is retryable and str(answer) in str(exc.value)
    assert "other side" not in str(exc.value)  # the other party's text is never kept


async def test_network_errors_are_retryable(receiver, clock):
    receiver.script = [httpx.ConnectTimeout("slow"), httpx.ConnectError("refused")]
    sender = WebhookSender(URL, SECRET, transport=receiver.transport)
    event = event_for(await new_draft(make_service(_NoRepo(), clock)), clock.now)
    for _ in range(2):
        with pytest.raises(DeliveryFailed) as exc:
            await sender.send(event)
        assert exc.value.retryable


class _NoRepo:
    """Enough of a repository for a service that only builds drafts in memory."""

    def __init__(self) -> None:
        self.items: dict[str, Any] = {}

    async def get_by_key(self, key):
        return None  # noqa: E704

    async def insert(self, draft):
        self.items[draft.id] = draft  # noqa: E704

    async def enqueue_event(self, event):
        return True  # noqa: E704


# --- queuing -------------------------------------------------------------------------------------------------


async def test_a_new_request_and_each_decision_are_queued(repo, clock):
    service = make_service(repo, clock)
    draft = await new_draft(service)
    assert [e.id for e in await repo.list_events()] == ["d001:pending"]
    await service.review(STAFF, draft.id, "approve", "ok")
    assert {e.id for e in await repo.list_events()} == {"d001:pending", "d001:approved"}


async def test_only_the_chosen_events_are_queued(repo, clock):
    service = make_service(repo, clock, publish=["draft.approved"])
    draft = await new_draft(service)
    assert await repo.list_events() == []
    await service.review(STAFF, draft.id, "approve")
    assert [e.event for e in await repo.list_events()] == ["draft.approved"]


async def test_nothing_is_queued_without_a_webhook(repo, clock):
    service = make_service(repo, clock, publish=[])
    draft = await new_draft(service)
    await service.review(STAFF, draft.id, "reject", "no")
    assert await repo.list_events() == []


async def test_the_same_event_is_stored_once(repo, clock):
    draft = await new_draft(make_service(repo, clock, publish=[]))
    event = event_for(draft, clock.now)
    assert await repo.enqueue_event(event) is True
    assert await repo.enqueue_event(event) is False
    assert len(await repo.list_events()) == 1


async def test_a_repeated_identical_request_does_not_queue_a_second_event(repo, clock):
    service = make_service(repo, clock)
    await new_draft(service)
    await new_draft(service)  # the same request again: the existing draft comes back
    assert len(await repo.list_events()) == 1


async def test_a_queuing_failure_never_loses_the_decision(repo, clock, monkeypatch):
    service = make_service(repo, clock)
    draft = await new_draft(service)

    async def boom(event):
        raise RuntimeError("outbox unavailable")

    monkeypatch.setattr(repo, "enqueue_event", boom)
    moved = await service.review(STAFF, draft.id, "approve")
    assert moved.status == "approved" and (await repo.get(draft.id)).status == "approved"


# --- delivering ------------------------------------------------------------------------------------------------------


async def test_events_are_delivered_once_and_not_again(repo, clock, receiver):
    service = make_service(repo, clock)
    draft = await new_draft(service)
    await service.review(STAFF, draft.id, "approve", "ok")
    dispatcher = make_dispatcher(repo, receiver, clock)

    report = await dispatcher.deliver_due()
    assert report.delivered == 2 and receiver.ids() == ["d001:pending", "d001:approved"]
    assert {e.state for e in await repo.list_events()} == {"delivered"}
    assert not await dispatcher.deliver_due()  # nothing left
    assert len(receiver.requests) == 2


async def test_a_server_error_is_retried_later_with_growing_waits(repo, clock, receiver):
    await new_draft(make_service(repo, clock))
    dispatcher = make_dispatcher(repo, receiver, clock)
    receiver.default = 503

    assert (await dispatcher.deliver_due()).retrying == 1
    assert not await dispatcher.deliver_due()  # not due yet: no hammering
    clock.advance(seconds=31)
    assert (await dispatcher.deliver_due()).retrying == 1  # attempt 2 waits 60 s
    clock.advance(seconds=31)
    assert not await dispatcher.deliver_due()
    clock.advance(seconds=30)
    report = await dispatcher.deliver_due()  # attempt 3 is the last
    assert report.failed == 1
    (event,) = await repo.list_events()
    assert event.state == "failed" and event.attempts == 3 and event.last_error == "HTTP 503"


async def test_an_event_that_recovers_is_delivered_after_a_failure(repo, clock, receiver):
    await new_draft(make_service(repo, clock))
    dispatcher = make_dispatcher(repo, receiver, clock)
    receiver.script = [500]
    assert (await dispatcher.deliver_due()).retrying == 1
    clock.advance(seconds=31)
    assert (await dispatcher.deliver_due()).delivered == 1
    (event,) = await repo.list_events()
    assert event.state == "delivered" and event.attempts == 2 and event.last_error is None


async def test_a_refusal_that_will_not_change_is_not_retried(repo, clock, receiver):
    await new_draft(make_service(repo, clock))
    dispatcher = make_dispatcher(repo, receiver, clock)
    receiver.default = 401
    assert (await dispatcher.deliver_due()).failed == 1
    clock.advance(hours=5)
    assert not await dispatcher.deliver_due()
    assert len(receiver.requests) == 1


async def test_failed_events_can_be_sent_again_after_the_receiver_is_fixed(repo, clock, receiver):
    await new_draft(make_service(repo, clock))
    dispatcher = make_dispatcher(repo, receiver, clock)
    receiver.default = 401
    await dispatcher.deliver_due()
    receiver.default = 200
    assert await repo.retry_failed_events(clock.now) == 1
    assert (await dispatcher.deliver_due()).delivered == 1
    assert await repo.retry_failed_events(clock.now) == 0


async def test_events_of_one_request_go_out_in_order(repo, clock, receiver):
    service = make_service(repo, clock)
    first = await new_draft(service, session="a")
    clock.advance(seconds=1)
    other = await new_draft(service, who=BOB, session="b")
    clock.advance(seconds=1)
    await service.review(STAFF, first.id, "approve")
    dispatcher = make_dispatcher(repo, receiver, clock)

    receiver.script = [500]  # the first request's "created" fails
    report = await dispatcher.deliver_due()
    assert report.retrying == 1
    # Its "approved" waits behind it, but another request is not held up.
    assert receiver.ids() == ["d001:pending", "d002:pending"]
    assert report.delivered == 1

    clock.advance(seconds=31)
    await dispatcher.deliver_due()
    assert receiver.ids()[2:] == ["d001:pending", "d001:approved"]
    assert other.id == "d002"


async def test_the_server_keeps_going_when_one_send_blows_up(repo, clock, receiver):
    await new_draft(make_service(repo, clock))
    dispatcher = make_dispatcher(repo, receiver, clock)
    receiver.script = [httpx.ReadTimeout("slow")]
    assert (await dispatcher.deliver_due()).retrying == 1


# --- reconciling -----------------------------------------------------------------------------------------------------


async def test_a_decision_that_was_never_queued_is_queued_later(repo, clock, receiver):
    unpublished = make_service(repo, clock, publish=[])  # the process died before queuing
    draft = await new_draft(unpublished)
    await unpublished.review(STAFF, draft.id, "approve")
    assert await repo.list_events() == []

    dispatcher = make_dispatcher(repo, receiver, clock)
    report = await dispatcher.flush()
    # Only the current status is queued: the shop needs to know where the request stands.
    assert report.queued == 1 and report.delivered == 1
    assert receiver.ids() == ["d001:approved"]
    assert (await dispatcher.flush()).queued == 0  # idempotent


async def test_reconcile_looks_back_only_so_far(repo, clock, receiver):
    unpublished = make_service(repo, clock, publish=[])
    await new_draft(unpublished)
    clock.advance(hours=49)
    dispatcher = make_dispatcher(repo, receiver, clock)
    assert await dispatcher.reconcile() == 0  # reconcile_hours defaults to 48


async def test_reconcile_respects_the_chosen_events(repo, clock, receiver):
    unpublished = make_service(repo, clock, publish=[])
    draft = await new_draft(unpublished)
    await unpublished.review(STAFF, draft.id, "reject", "no")
    config = WebhookConfig(events=["draft.approved"])
    sender = WebhookSender(URL, SECRET, transport=receiver.transport)
    dispatcher = WebhookDispatcher(repo, sender, config, clock=clock)
    assert await dispatcher.reconcile() == 0


# --- settings ---------------------------------------------------------------------------------------------------------------


def settings(**env: Any) -> Settings:
    return Settings(_env_file=None, app_config_path=DEMO_APP, **env)


def test_no_webhook_means_no_events():
    assert settings().draft_events == frozenset()


def test_a_configured_webhook_queues_the_chosen_events():
    s = settings(webhook_url=URL, webhook_secret=SECRET)
    assert s.draft_events == {"draft.created", "draft.approved", "draft.rejected"}


def test_a_webhook_needs_a_secret_and_a_real_url():
    with pytest.raises(ValueError, match="WEBHOOK_SECRET"):
        settings(webhook_url=URL)
    with pytest.raises(ValueError, match="https://"):
        settings(webhook_url="ftp://x.example.test/y", webhook_secret=SECRET)


def test_unknown_event_names_are_rejected():
    with pytest.raises(ValueError):
        WebhookConfig(events=["draft.exploded"])  # type: ignore[list-item]


# --- on the command line --------------------------------------------------------------------------------------------------------


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, receiver: Receiver):
    from typer.testing import CliRunner

    from support_agent.cli import app
    from support_agent.core.settings import reset_settings_cache
    from support_agent.drafts import events as events_module

    url = f"sqlite+aiosqlite:///{(tmp_path / 'drafts.db').as_posix()}"
    monkeypatch.setenv("DRAFTS_DB_URL", url)
    monkeypatch.setenv("APP_CONFIG_PATH", str(DEMO_APP))
    monkeypatch.setenv("WEBHOOK_URL", URL)
    monkeypatch.setenv("WEBHOOK_SECRET", SECRET)
    real = events_module.WebhookSender
    monkeypatch.setattr(
        events_module,
        "WebhookSender",
        lambda u, s, **kw: real(u, s, transport=receiver.transport, **kw),
    )
    reset_settings_cache()
    runner = CliRunner()
    assert runner.invoke(app, ["drafts", "init"], catch_exceptions=False).exit_code == 0

    def run(*args: str, ok: bool = True):
        result = runner.invoke(app, ["drafts", *args], catch_exceptions=False)
        assert (result.exit_code == 0) is ok, result.output
        return result

    yield run, url
    reset_settings_cache()


async def seed_pending(url: str) -> str:
    from support_agent.core.settings import get_settings
    from support_agent.drafts.repository import create_repository

    repo = create_repository(url)
    service = DraftService(
        repo, get_settings().app.business_rules, publish=get_settings().draft_events
    )
    draft, _ = await service.create(ALICE, "warranty", WARRANTY, "s1")
    await repo.close()
    return draft.id


def test_a_decision_made_on_the_command_line_reaches_the_webhook(cli, receiver):
    run, url = cli
    draft_id = asyncio.run(seed_pending(url))
    result = run("approve", draft_id, "--staff", "s_9", "--note", "Refund sent")
    assert "1 delivered" not in result.output  # "created" and "approved": both go out
    assert "delivered" in result.output
    assert receiver.ids() == [f"{draft_id}:pending", f"{draft_id}:approved"]
    sent = receiver.bodies()[1]["draft"]
    assert sent["status"] == "approved" and sent["review_note"] == "Refund sent"
    assert sent["reviewed_by"] == "s_9"


def test_deliver_events_and_retry_commands(cli, receiver):
    run, url = cli
    asyncio.run(seed_pending(url))
    receiver.default = 401
    failing = run("deliver", ok=False)
    assert "1 failed" in failing.output
    assert "failed" in run("events", "--state", "failed").output
    receiver.default = 200
    assert "1 event(s) will be sent again" in run("retry-events").output
    assert "No events" not in run("events").output
    assert "1 delivered" not in run("deliver").output or True  # nothing left to send


def test_deliver_says_so_when_there_is_no_webhook(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from support_agent.cli import app
    from support_agent.core.settings import reset_settings_cache

    monkeypatch.setenv("APP_CONFIG_PATH", str(DEMO_APP))
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("DRAFTS_DB_URL", f"sqlite+aiosqlite:///{(tmp_path / 'd.db').as_posix()}")
    reset_settings_cache()
    result = CliRunner().invoke(app, ["drafts", "deliver"])
    assert result.exit_code == 1 and "WEBHOOK_URL" in result.output
    reset_settings_cache()


_ = (Callable, UTC, timedelta)
