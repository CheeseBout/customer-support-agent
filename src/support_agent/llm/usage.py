"""Token usage accounting via a LangChain callback."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class UsageTracker(BaseCallbackHandler):
    """Sums provider-reported usage over every LLM call it is attached to."""

    def __init__(self) -> None:
        self.usage = Usage()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    self.usage.input_tokens += int(meta.get("input_tokens", 0))
                    self.usage.output_tokens += int(meta.get("output_tokens", 0))
