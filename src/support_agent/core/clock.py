"""A fixed "now", for evaluating against a snapshot of a shop's data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

NOW_ENV = "SUPPORT_AGENT_NOW"


def parse_instant(text: str, timezone: str = "Asia/Ho_Chi_Minh") -> datetime:
    """An ISO 8601 moment as UTC. Without a zone it is read in `timezone` (the business zone)."""
    moment = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=ZoneInfo(timezone))
    return moment.astimezone(UTC)


def frozen_clock(text: str, timezone: str = "Asia/Ho_Chi_Minh") -> Callable[[], datetime]:
    moment = parse_instant(text, timezone)
    return lambda: moment
