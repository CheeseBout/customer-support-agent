"""Optional Langfuse tracing. Everything here is a no-op unless both keys are configured."""

from __future__ import annotations

import logging
from typing import Any

from support_agent.core.settings import Settings

log = logging.getLogger(__name__)


class Tracing:
    """Thin wrapper so callers never import Langfuse directly or branch on availability."""

    def __init__(self, client: Any = None) -> None:
        self._client = client

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def trace_id(self, seed: str) -> str | None:
        """A deterministic trace id, so a run can be linked to its scores before it finishes."""
        return self._client.create_trace_id(seed=seed) if self._client else None

    def handler(self, trace_id: str | None) -> Any | None:
        """A LangChain callback handler that records the run under `trace_id`."""
        if not self._client:
            return None
        from langfuse.langchain import CallbackHandler

        # `Any`: whether langfuse is installed (an optional extra) must not change type checking.
        context: Any = {"trace_id": trace_id} if trace_id else None
        return CallbackHandler(trace_context=context)

    def score(
        self, trace_id: str | None, name: str, value: float, *, comment: str | None = None
    ) -> None:
        if self._client and trace_id:
            self._client.create_score(
                name=name,
                value=float(value),
                trace_id=trace_id,
                data_type="NUMERIC",
                comment=comment,
            )

    def flush(self) -> None:
        if self._client:
            self._client.flush()


def create_tracing(settings: Settings) -> Tracing:
    """Enabled only with LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY and the optional package."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return Tracing()
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("Langfuse keys are set but the package is missing: pip install '.[langfuse]'")
        return Tracing()
    client = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        host=settings.langfuse_host,
    )
    return Tracing(client)
