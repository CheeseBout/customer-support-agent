"""`support-agent drafts ...`: the staff side of customer requests (SPEC 10, FR-204)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from support_agent.core.principal import Principal
from support_agent.core.settings import Settings, get_settings
from support_agent.drafts.events import DeliveryReport, build_dispatcher
from support_agent.drafts.models import Draft
from support_agent.drafts.repository import create_repository
from support_agent.drafts.service import DraftError, DraftService

drafts_app = typer.Typer(help="Customer requests waiting for staff (refund, return, ...).")
console = Console()

STAFF = Annotated[str, typer.Option("--staff", help="Staff member id (stand-in for the login)")]
URL = Annotated[str | None, typer.Option("--url", help="Drafts DB URL (default: DRAFTS_DB_URL)")]


def _url(settings: Settings, url: str | None) -> str:
    found = url or settings.drafts_db_url
    if not found:
        console.print("[red]DRAFTS_DB_URL is not set (see .env.example).[/red]")
        raise typer.Exit(1)
    return found


def _run[T](
    action: Callable[[DraftService], Awaitable[T]],
    url: str | None,
    *,
    create_schema: bool = False,
    deliver: bool = False,
) -> T:
    """Run `action` on the drafts. With `deliver`, then send the events it queued to the shop
    webhook (if there is one), so a decision made here is not left waiting for the server."""
    settings = get_settings()

    async def go() -> T:
        repo = create_repository(_url(settings, url))
        try:
            if create_schema:
                await repo.create_schema()
            service = DraftService(repo, settings.app.business_rules, publish=settings.draft_events)
            result = await action(service)
            if deliver and (dispatcher := build_dispatcher(settings, repo)) is not None:
                try:
                    report = await dispatcher.flush()
                    if report:
                        console.print(f"[dim]webhook: {_report(report)}[/dim]")
                finally:
                    await dispatcher.close()
            return result
        finally:
            await repo.close()

    try:
        return asyncio.run(go())
    except DraftError as exc:
        console.print(f"[red]{exc.code}: {exc.message}[/red]")
        raise typer.Exit(1) from exc


def _report(report: DeliveryReport) -> str:
    parts = [
        f"{report.delivered} delivered",
        f"{report.retrying} will be retried",
        f"{report.failed} failed",
    ]
    return ", ".join(parts)


def _staff(staff: str) -> Principal:
    return Principal(user_id=staff, role="staff")


def _describe(draft: Draft) -> str:
    p = draft.payload
    if draft.type in ("refund", "return"):
        return (
            f"order {p.get('order_id')} - {p.get('refundable_amount', '')} ({p.get('reason_code')})"
        )
    if draft.type == "warranty":
        return f"order {p.get('order_id')} {p.get('sku')} - {p.get('issue_description')}"
    if draft.type == "handoff":
        contact = f" ({p['contact']})" if p.get("contact") else ""
        order = f"order {p['order_id']}: " if p.get("order_id") else ""
        return f"needs a person{contact} - {order}{p.get('reason')}"
    return f"{p.get('total')} {p.get('currency')} to {p.get('shipping_address')}"


@drafts_app.command("init")
def init_schema(url: URL = None) -> None:
    """Create the support_drafts table. Use an account that is allowed to create tables."""

    async def go(_: DraftService) -> None:
        return None

    _run(go, url, create_schema=True)
    console.print("[green]support_drafts is ready.[/green]")


@drafts_app.command("list")
def list_drafts(
    staff: STAFF = "s_1",
    status: Annotated[str, typer.Option(help="pending | approved | rejected | cancelled | all")] = (
        "pending"
    ),
    draft_type: Annotated[
        str | None, typer.Option("--type", help="refund|return|warranty|order|handoff")
    ] = None,
    url: URL = None,
) -> None:
    """The review queue, oldest first. Large refunds are marked."""
    statuses = None if status == "all" else (status,)
    found = _run(lambda s: s.queue(_staff(staff), statuses=statuses, draft_type=draft_type), url)
    table = Table()
    table.add_column("id", no_wrap=True)  # the id must stay whole so it can be copied
    for column in ("type", "customer", "status", "priority", "request"):
        table.add_column(column)
    for d in found:
        table.add_row(
            d.id,
            d.type,
            d.customer_id,
            d.status,
            "PRIORITY" if d.priority_review else "",
            _describe(d),
        )
    console.print(table if found else "[dim]Nothing in the queue.[/dim]")


@drafts_app.command("show")
def show(draft_id: str, staff: STAFF = "s_1", url: URL = None) -> None:
    """Everything about one request, including the rule checks it was made under."""
    draft = _run(lambda s: s.get_for(_staff(staff), draft_id), url)
    console.print_json(draft.model_dump_json())


@drafts_app.command("approve")
def approve(
    draft_id: str,
    staff: STAFF = "s_1",
    note: Annotated[str | None, typer.Option(help="Optional note for the customer")] = None,
    url: URL = None,
) -> None:
    """Approve a pending request. This records the decision; fulfilment happens in the shop."""
    moved = _run(lambda s: s.review(_staff(staff), draft_id, "approve", note), url, deliver=True)
    console.print(f"[green]{moved.id} approved by {moved.reviewed_by}.[/green]")


@drafts_app.command("reject")
def reject(
    draft_id: str,
    note: Annotated[str, typer.Option(help="Why: the customer is told this")],
    staff: STAFF = "s_1",
    url: URL = None,
) -> None:
    """Reject a pending request. A reason is required."""
    moved = _run(lambda s: s.review(_staff(staff), draft_id, "reject", note), url, deliver=True)
    console.print(f"[yellow]{moved.id} rejected by {moved.reviewed_by}.[/yellow]")


@drafts_app.command("cancel")
def cancel(draft_id: str, staff: STAFF = "s_1", url: URL = None) -> None:
    """Withdraw a pending request."""
    moved = _run(lambda s: s.cancel(_staff(staff), draft_id), url, deliver=True)
    console.print(f"{moved.id} cancelled.")


@drafts_app.command("deliver")
def deliver(url: URL = None) -> None:
    """Send the queued events to the shop webhook now (the server also does this by itself)."""
    settings = get_settings()
    if not settings.webhook_url:
        console.print("[red]WEBHOOK_URL is not set: there is no webhook to deliver to.[/red]")
        raise typer.Exit(1)

    async def go(service: DraftService) -> DeliveryReport:
        dispatcher = build_dispatcher(settings, service.repo)
        assert dispatcher is not None
        try:
            return await dispatcher.flush()
        finally:
            await dispatcher.close()

    report = _run(go, url)
    console.print(_report(report) + (f", {report.queued} queued again" if report.queued else ""))
    if report.failed:
        raise typer.Exit(1)


@drafts_app.command("events")
def events(
    state: Annotated[str | None, typer.Option(help="pending | delivered | failed")] = None,
    limit: Annotated[int, typer.Option(help="How many, newest first")] = 20,
    url: URL = None,
) -> None:
    """What was queued for the shop webhook, and how delivery is going."""
    found = _run(lambda s: s.repo.list_events(state=state, limit=limit), url)
    for e in found:  # one line each: the ids are long and a table would squeeze everything else
        error = f"  {e.last_error}" if e.last_error else ""
        console.print(
            f"{e.state:<9} {e.event:<15} tries={e.attempts}  draft {e.draft_id}{error}",
            soft_wrap=True,
            markup=False,
        )
    if not found:
        console.print("[dim]No events.[/dim]")


@drafts_app.command("retry-events")
def retry_events(url: URL = None) -> None:
    """Try again every event that was given up on (after fixing the receiving side)."""

    async def go(service: DraftService) -> int:
        return await service.repo.retry_failed_events(datetime.now(UTC))

    count = _run(go, url, deliver=True)
    console.print(f"{count} event(s) will be sent again.")


def summary_lines(stop: dict[str, Any]) -> list[str]:
    """How a confirmation is shown on a terminal."""
    lines = [f"Please confirm this {stop['draft_type']} request:"]
    lines += [f"  {key}: {value}" for key, value in stop["summary"].items()]
    if stop.get("priority_review"):
        lines.append("  (flagged for priority review because of its amount)")
    lines.append(f"  valid until {stop['expires_at']}")
    return lines
