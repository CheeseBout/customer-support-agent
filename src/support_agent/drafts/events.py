"""Telling the shop's own system what happened to a request (the webhook connector).

Approving a request only records the decision: refunding the money or creating the order is the
shop's job. This module is how the shop's system finds out, without anyone copying decisions by
hand.

Delivery is built so that a decision is never lost and never sent twice by mistake:

* An event is written to the drafts database (the *outbox*) when the request is made or decided.
  Sending happens afterwards and can fail, wait and try again; the decision itself is already saved.
* The event id is `<draft id>:<status>`, so queuing the same event twice stores it once, and the
  shop's system can use the id to ignore a repeat.
* Every request is signed (`X-Support-Signature: t=<unix time>,v1=<hex>`, HMAC-SHA256 over
  `<t>.<body>`), so the shop's system can tell it comes from this service and is not a replay.
* Events of one request go out in order: while one is waiting to be retried, later ones for the
  same request wait too.
* A status that was saved but never queued (the process died in between) is queued again by
  `reconcile`.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel

from support_agent.core.settings import Settings, WebhookConfig
from support_agent.drafts.models import Draft

if TYPE_CHECKING:
    from support_agent.drafts.repository import DraftRepository

log = logging.getLogger(__name__)

EVENT_NAMES = {
    "pending": "draft.created",
    "approved": "draft.approved",
    "rejected": "draft.rejected",
    "cancelled": "draft.cancelled",
}
MAX_BACKOFF_SECONDS = 3600.0
RECONCILE_EVERY_SECONDS = 600.0
BATCH = 50
EventState = Literal["pending", "delivered", "failed"]


class DraftEvent(BaseModel):
    id: str  # "<draft id>:<status>": the same event is stored once
    draft_id: str
    event: str
    body: dict[str, Any]  # what the shop's system receives
    state: EventState = "pending"
    attempts: int = 0
    next_attempt_at: datetime
    last_error: str | None = None
    created_at: datetime
    delivered_at: datetime | None = None


def _public(draft: Draft) -> dict[str, Any]:
    """The draft as the shop's system sees it. No idempotency key, no session id."""
    return {
        "id": draft.id,
        "type": draft.type,
        "status": draft.status,
        "customer_id": draft.customer_id,
        "order_id": draft.order_id,
        "payload": draft.payload,
        "priority_review": draft.priority_review,
        "reviewed_by": draft.reviewed_by,
        "reviewed_at": draft.reviewed_at.isoformat() if draft.reviewed_at else None,
        "review_note": draft.review_note,
        "created_at": draft.created_at.isoformat(),
        "updated_at": draft.updated_at.isoformat(),
    }


def event_for(draft: Draft, now: datetime) -> DraftEvent:
    """The event for the draft's current status, with a snapshot of the draft."""
    name = EVENT_NAMES[draft.status]
    event_id = f"{draft.id}:{draft.status}"
    return DraftEvent(
        id=event_id,
        draft_id=draft.id,
        event=name,
        body={
            "id": event_id,
            "event": name,
            "created_at": now.isoformat(),
            "draft": _public(draft),
        },
        next_attempt_at=now,
        created_at=now,
    )


# --- signing --------------------------------------------------------------------------------------


def sign(secret: str, timestamp: int, body: bytes) -> str:
    return hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def verify_signature(
    secret: str, header: str, body: bytes, *, tolerance_seconds: int = 300, now: float | None = None
) -> bool:
    """What the receiving side runs: is this signed with our secret, and recent enough?

    Included here so the shop's developers have a reference to copy, and so the format is tested
    from both sides.
    """
    try:
        parts = dict(item.split("=", 1) for item in header.split(","))
        timestamp = int(parts["t"])
        given = parts["v1"]
    except (KeyError, ValueError):
        return False
    if abs((time.time() if now is None else now) - timestamp) > tolerance_seconds:
        return False
    return hmac.compare_digest(sign(secret, timestamp, body), given)


# --- sending --------------------------------------------------------------------------------------


class DeliveryFailed(Exception):
    """The shop system did not accept the event; `retryable`: can trying again help?"""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class WebhookSender:
    def __init__(
        self,
        url: str,
        secret: str,
        *,
        timeout_seconds: float = 10,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.url = url
        self._secret = secret
        self._clock = clock
        # No redirects: a redirect could send a signed body to a place nobody configured.
        self._client = httpx.AsyncClient(
            timeout=timeout_seconds, follow_redirects=False, transport=transport
        )

    async def send(self, event: DraftEvent) -> None:
        body = json.dumps(event.body, ensure_ascii=False, separators=(",", ":")).encode()
        stamp = int(self._clock())
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "support-agent-webhook/1",
            "X-Support-Event": event.event,
            "X-Support-Event-Id": event.id,
            "X-Support-Signature": f"t={stamp},v1={sign(self._secret, stamp, body)}",
        }
        try:
            response = await self._client.post(self.url, content=body, headers=headers)
        except httpx.HTTPError as exc:  # timeouts, refused connections, TLS errors
            raise DeliveryFailed(type(exc).__name__, retryable=True) from exc
        code = response.status_code
        if 200 <= code < 300:
            return
        # Never log the response body: it is the other party's text.
        retryable = code >= 500 or code in (408, 425, 429)
        raise DeliveryFailed(f"HTTP {code}", retryable=retryable)

    async def close(self) -> None:
        await self._client.aclose()


# --- the dispatcher ------------------------------------------------------------------------------


class DeliveryReport(BaseModel):
    delivered: int = 0
    retrying: int = 0
    failed: int = 0
    queued: int = 0  # events added by reconcile

    def __bool__(self) -> bool:
        return bool(self.delivered or self.retrying or self.failed or self.queued)


def _send_order(event: DraftEvent) -> tuple[datetime, int, str]:
    """Oldest first; events stored in the same instant (a clock or column that cannot tell them
    apart) go out as they happened: a request is created before it is decided."""
    return event.created_at, 0 if event.event == EVENT_NAMES["pending"] else 1, event.id


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WebhookDispatcher:
    def __init__(
        self,
        repo: DraftRepository,
        sender: WebhookSender,
        config: WebhookConfig,
        *,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.repo = repo
        self.sender = sender
        self.cfg = config
        self._clock = clock

    async def reconcile(self) -> int:
        """Queue the event of any recent request whose current status was never queued."""
        now = self._clock()
        since = now - timedelta(hours=self.cfg.reconcile_hours)
        wanted = set(self.cfg.events)
        added = 0
        for draft in await self.repo.list(updated_after=since, limit=500):
            if EVENT_NAMES[draft.status] in wanted and await self.repo.enqueue_event(
                event_for(draft, now)
            ):
                added += 1
        return added

    async def deliver_due(self) -> DeliveryReport:
        """Send every event that is due. Returns what happened; never raises for a failed send."""
        report = DeliveryReport()
        now = self._clock()
        waiting: set[str] = set()  # drafts with an earlier event still to be retried
        for event in sorted(await self.repo.due_events(now, BATCH), key=_send_order):
            if event.draft_id in waiting:
                continue
            attempts = event.attempts + 1
            try:
                await self.sender.send(event)
            except DeliveryFailed as exc:
                final = not exc.retryable or attempts >= self.cfg.max_attempts
                delay = min(self.cfg.backoff_seconds * 2 ** (attempts - 1), MAX_BACKOFF_SECONDS)
                await self.repo.update_event(
                    event.id,
                    state="failed" if final else "pending",
                    attempts=attempts,
                    next_attempt_at=now + timedelta(seconds=delay),
                    last_error=str(exc),
                    delivered_at=None,
                )
                waiting.add(event.draft_id)
                if final:
                    report.failed += 1
                    log.error("webhook event %s given up on: %s", event.id, exc)
                else:
                    report.retrying += 1
                    log.warning("webhook event %s will be retried: %s", event.id, exc)
                continue
            await self.repo.update_event(
                event.id,
                state="delivered",
                attempts=attempts,
                next_attempt_at=now,
                last_error=None,
                delivered_at=self._clock(),
            )
            report.delivered += 1
        return report

    async def flush(self) -> DeliveryReport:
        """Reconcile, then send what is due: what the CLI and the periodic loop both do."""
        queued = await self.reconcile()
        report = await self.deliver_due()
        report.queued = queued
        return report

    async def run_forever(self) -> None:
        last_reconcile = 0.0
        while True:
            try:
                if (
                    time.monotonic() - last_reconcile >= RECONCILE_EVERY_SECONDS
                    or not last_reconcile
                ):
                    await self.reconcile()
                    last_reconcile = time.monotonic()
                await self.deliver_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("webhook delivery loop failed; trying again")
            await asyncio.sleep(self.cfg.poll_seconds)

    async def close(self) -> None:
        await self.sender.close()


def build_dispatcher(settings: Settings, repo: DraftRepository) -> WebhookDispatcher | None:
    """The dispatcher for this deployment, or None when no webhook is configured."""
    if not settings.webhook_url or settings.webhook_secret is None:
        return None
    cfg = settings.app.connectors.webhook
    sender = WebhookSender(
        settings.webhook_url,
        settings.webhook_secret.get_secret_value(),
        timeout_seconds=cfg.timeout_seconds,
    )
    return WebhookDispatcher(repo, sender, cfg)


__all__ = [
    "EVENT_NAMES",
    "DeliveryFailed",
    "DeliveryReport",
    "DraftEvent",
    "WebhookDispatcher",
    "WebhookSender",
    "build_dispatcher",
    "event_for",
    "sign",
    "verify_signature",
]
