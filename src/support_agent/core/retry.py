"""Retry with exponential backoff (SPEC NFR-005: at most 2 retries, then a friendly error)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


async def with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    base_delay: float,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    retry_if: Callable[[T], bool] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Call `fn`, retrying on listed exceptions or when `retry_if(result)` is true.

    Delays are `base_delay * 2**attempt`. The last attempt's exception propagates and its
    result is returned as-is, so callers can still inspect a final failed result.
    """
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            result = await fn()
        except retry_exceptions as exc:
            if last:
                raise
            log.warning("attempt %d failed (%s); retrying", attempt + 1, type(exc).__name__)
        else:
            if last or retry_if is None or not retry_if(result):
                return result
            log.warning("attempt %d returned a retryable result; retrying", attempt + 1)
        await sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")  # pragma: no cover
