"""Draft lifecycle rules: who may do what, in which state (SPEC 10.3, 10.4, FR-203/204)."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import ValidationError

from support_agent.core.principal import Principal
from support_agent.core.results import ErrorCode
from support_agent.core.settings import BusinessRules
from support_agent.drafts.models import (
    ACTIVE_STATUSES,
    PAYLOAD_MODELS,
    TRANSITIONS,
    Draft,
    idempotency_key,
)
from support_agent.drafts.repository import DraftRepository, DuplicateKey
from support_agent.rules.warranty import needs_priority_review

Decision = Literal["approve", "reject"]


class DraftError(Exception):
    """A refused draft operation. `code`: NOT_FOUND, FORBIDDEN, CONFLICT or INVALID_ARGUMENT."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _utcnow() -> datetime:
    return datetime.now(UTC)


class DraftService:
    def __init__(
        self,
        repo: DraftRepository,
        rules: BusinessRules,
        *,
        clock: Callable[[], datetime] = _utcnow,
        new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self.repo = repo
        self.rules = rules
        self._clock = clock
        self._new_id = new_id

    # --- customer side ---------------------------------------------------------------
    async def create(
        self,
        principal: Principal,
        draft_type: str,
        payload: dict[str, Any],
        session_id: str,
        *,
        key: str | None = None,
    ) -> tuple[Draft, bool]:
        """Create a `pending` draft for the caller. Returns `(draft, created)`.

        Idempotent: repeating the same request returns the existing draft with `created=False`
        instead of making a second one (NFR-006).
        """
        model = PAYLOAD_MODELS.get(draft_type)
        if model is None:
            raise DraftError("INVALID_ARGUMENT", f"Unknown draft type {draft_type!r}.")
        try:
            clean = model.model_validate(payload).model_dump(mode="json")
        except ValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            raise DraftError(
                "INVALID_ARGUMENT", f"Invalid {draft_type} request: {fields}."
            ) from exc

        key = key or idempotency_key(principal.user_id, session_id, draft_type, clean)
        existing = await self.repo.get_by_key(key)
        if existing is not None:
            return existing, False

        now = self._clock()
        draft = Draft(
            id=self._new_id(),
            type=draft_type,  # type: ignore[arg-type]
            customer_id=principal.user_id,
            session_id=session_id,
            status="pending",
            payload=clean,
            idempotency_key=key,
            # SPEC 9.4: a large refund is flagged for senior review. Nothing is auto-approved.
            priority_review=draft_type == "refund"
            and needs_priority_review(clean["refundable_amount"], self.rules.refund),
            created_at=now,
            updated_at=now,
        )
        try:
            await self.repo.insert(draft)
        except DuplicateKey:  # a concurrent identical request won the race
            winner = await self.repo.get_by_key(key)
            if winner is not None:
                return winner, False
            raise
        return draft, True

    async def list_for(
        self, principal: Principal, *, statuses: Sequence[str] | None = None, limit: int = 10
    ) -> list[Draft]:
        """The caller's own drafts, newest first."""
        return await self.repo.list(
            customer_id=principal.user_id, statuses=statuses, limit=max(1, min(limit, 50))
        )

    async def get_for(self, principal: Principal, draft_id: str) -> Draft:
        """A draft the caller may see. Someone else's looks exactly like a missing one."""
        draft = await self.repo.get(draft_id)
        if draft is None or (principal.role != "staff" and draft.customer_id != principal.user_id):
            raise DraftError("NOT_FOUND", f"Draft {draft_id!r} was not found.")
        return draft

    async def cancel(self, principal: Principal, draft_id: str) -> Draft:
        """The owner (or staff) withdraws a request that is still pending."""
        draft = await self.get_for(principal, draft_id)
        return await self._move(draft, "cancelled", principal)

    # --- staff side ------------------------------------------------------------------
    async def review(
        self, principal: Principal, draft_id: str, decision: Decision, note: str | None = None
    ) -> Draft:
        """Staff approve or reject. Rejecting needs a note the customer can be told."""
        if principal.role != "staff":
            raise DraftError("FORBIDDEN", "Only staff can review drafts.")
        if decision == "reject" and not (note and note.strip()):
            raise DraftError("INVALID_ARGUMENT", "A note is required when rejecting a draft.")
        draft = await self.get_for(principal, draft_id)
        target = "approved" if decision == "approve" else "rejected"
        return await self._move(draft, target, principal, note=note)

    async def queue(
        self,
        principal: Principal,
        *,
        statuses: Sequence[str] | None = ("pending",),
        draft_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Draft]:
        """The review queue, oldest first, with large refunds flagged by `priority_review`."""
        if principal.role != "staff":
            raise DraftError("FORBIDDEN", "Only staff can see the review queue.")
        return await self.repo.list(
            statuses=statuses, draft_type=draft_type, order="oldest", limit=limit, offset=offset
        )

    # --- used by the agent's own checks ----------------------------------------------
    async def existing_requests(self, customer_id: str, order_id: str) -> list[dict[str, Any]]:
        """Pending/approved return or refund drafts, shaped like the shop's own request rows.

        Feeds the "no active request for the same item" rule (SPEC 9.1, check 5), so asking
        twice through the agent cannot stack two refunds on one product.
        """
        found = await self.repo.list(
            customer_id=customer_id, statuses=ACTIVE_STATUSES, order_id=order_id, limit=50
        )
        rows: list[dict[str, Any]] = []
        for draft in found:
            if draft.type in ("return", "refund"):
                rows += [{"status": draft.status, "sku": sku} for sku in sorted(draft.sku_set())]
        return rows

    # --- internals -------------------------------------------------------------------
    async def _move(
        self, draft: Draft, target: str, principal: Principal, *, note: str | None = None
    ) -> Draft:
        if target not in TRANSITIONS[draft.status]:
            raise DraftError("CONFLICT", f"A {draft.status} draft cannot become {target}.")
        moved = await self.repo.transition(
            draft.id,
            expected=draft.status,
            to=target,
            now=self._clock(),
            reviewed_by=principal.user_id if target in ("approved", "rejected") else None,
            note=note,
        )
        if moved is None:  # someone else changed it between our read and our write
            raise DraftError("CONFLICT", "The draft was changed by someone else; reload and retry.")
        return moved
