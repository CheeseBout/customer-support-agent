"""Test doubles: scripted chat model, deterministic hashing embeddings, offline sparse encoder."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Callable
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import ConfigDict, Field
from qdrant_client import models

from support_agent.llm.structured import extract_json, message_text
from support_agent.mcp_db.service import fold

Responder = Callable[[list[BaseMessage], list[Any]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """A chat model whose replies come from `responder(messages, bound_tools)`.

    Records every call in `calls` so tests can assert on what the model was shown.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Any = None
    tools: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    # One entry per bind_tools(): {"tools": [names], "kwargs": {...}}, shared by all copies.
    bind_log: list[dict[str, Any]] = Field(default_factory=list)
    stream_chunk_chars: int = 4  # text is streamed in pieces this long, to exercise chunking

    @property
    def _llm_type(self) -> str:
        return "scripted"

    async def _astream(  # type: ignore[override]
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(list(messages))
        ai = self.responder(messages, self.tools)
        delay = float(ai.additional_kwargs.get("delay", 0))
        if delay:
            await asyncio.sleep(delay)  # lets a test exercise timeouts and cancellation
        if ai.tool_calls:
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                    for i, c in enumerate(ai.tool_calls)
                ],
                usage_metadata=ai.usage_metadata,
            )
            yield ChatGenerationChunk(message=chunk)
            return
        text, n = str(ai.content), max(1, self.stream_chunk_chars)
        pieces = [text[i : i + n] for i in range(0, len(text), n)] or [""]
        for i, piece in enumerate(pieces):
            usage = ai.usage_metadata if i == len(pieces) - 1 else None
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece, usage_metadata=usage))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        ai = self.responder(messages, self.tools)
        return ChatResult(generations=[ChatGeneration(message=ai)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, None, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:  # type: ignore[override]
        bound = self.model_copy(update={"tools": list(tools)})
        bound.calls = self.calls  # share the call logs with the parent
        bound.bind_log = self.bind_log
        self.bind_log.append(
            {"tools": [getattr(t, "name", str(t)) for t in tools], "kwargs": kwargs}
        )
        return bound

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        async def run(messages: Any, config: Any = None) -> Any:
            ai = await self.ainvoke(messages, config)
            return schema.model_validate(extract_json(message_text(ai)))

        return RunnableLambda(run)

    # --- convenience --------------------------------------------------------------------
    def system_prompts(self) -> list[str]:
        return [str(m.content) for call in self.calls for m in call if isinstance(m, SystemMessage)]

    def last_human_text(self) -> str:
        for call in reversed(self.calls):
            for m in reversed(call):
                if m.type == "human":
                    return str(m.content)
        return ""


def ai_json(payload: dict[str, Any], *, tokens: tuple[int, int] = (10, 5)) -> AIMessage:
    return AIMessage(
        content=json.dumps(payload, ensure_ascii=False),
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )


class SupportFakeLLM:
    """Builds a ScriptedChatModel that plays router, tool-picker and answerer.

    Tasks are recognised by the system prompt, the same way a real model would be steered.
    """

    def __init__(
        self,
        *,
        route: str = "policy",
        confidence: float = 0.95,
        tool_plan: list[dict[str, Any]] | None = None,
        answer: str = "FAKE ANSWER",
        sufficient: bool = True,
        used_sources: list[int] | None = None,
        route_error: Exception | None = None,
        answer_error: Exception | None = None,
    ) -> None:
        self.route = route
        self.confidence = confidence
        self.tool_plan = tool_plan or []
        self.answer = answer
        self.sufficient = sufficient
        self.used_sources = [1] if used_sources is None else used_sources
        self.route_error = route_error
        self.answer_error = answer_error
        self._tool_round = 0
        self.model = ScriptedChatModel(responder=self._respond)

    def _respond(self, messages: list[BaseMessage], tools: list[Any]) -> AIMessage:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        if system.startswith("Classify the customer's message"):
            if self.route_error:
                raise self.route_error
            return ai_json({"route": self.route, "confidence": self.confidence})
        if system.startswith("You gather facts"):
            if self._tool_round < len(self.tool_plan):
                call = self.tool_plan[self._tool_round]
                self._tool_round += 1
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": call["name"],
                            "args": call["args"],
                            "id": f"call{self._tool_round}",
                        }
                    ],
                )
            return AIMessage(content="done")
        if system.startswith("You are the customer support assistant"):
            if self.answer_error:
                raise self.answer_error
            return ai_json(
                {
                    "sufficient": self.sufficient,
                    "answer": self.answer,
                    "used_sources": self.used_sources if self.sufficient else [],
                }
            )
        return AIMessage(content="ok")


class AgentFakeLLM:
    """Plays the router and the agent loop from a script.

    `steps` are consumed one per agent-model call, across turns:
      {"tools": [(name, args), ...]}  the model asks for these tool calls
      {"text": "..."}                 the model writes its final answer
      {"error": Exception}            the model call fails
    """

    def __init__(
        self,
        steps: list[dict[str, Any]],
        *,
        route: str = "policy",
        confidence: float = 0.95,
        route_error: Exception | None = None,
        chunk_chars: int = 3,
    ) -> None:
        self.steps = list(steps)
        self.route = route
        self.confidence = confidence
        self.route_error = route_error
        self._next = 0
        self._call_ids = 0
        self.model = ScriptedChatModel(responder=self._respond, stream_chunk_chars=chunk_chars)

    def add_steps(self, *steps: dict[str, Any]) -> None:
        self.steps.extend(steps)

    def _respond(self, messages: list[BaseMessage], tools: list[Any]) -> AIMessage:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        if system.startswith("Classify the customer's message"):
            if self.route_error:
                raise self.route_error
            return ai_json({"route": self.route, "confidence": self.confidence})
        if not system.startswith("You are the customer support agent"):
            return AIMessage(content="ok")
        if self._next >= len(self.steps):
            raise AssertionError("the agent asked the model for more steps than the test scripted")
        step = self.steps[self._next]
        self._next += 1
        usage = {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}
        extra = {"delay": step["delay"]} if "delay" in step else {}
        if "error" in step:
            raise step["error"]
        if "tools" in step:
            calls = []
            for name, args in step["tools"]:
                self._call_ids += 1
                calls.append({"name": name, "args": args, "id": f"call{self._call_ids}"})
            return AIMessage(
                content="", tool_calls=calls, usage_metadata=usage, additional_kwargs=extra
            )
        return AIMessage(content=step["text"], usage_metadata=usage, additional_kwargs=extra)

    # --- what the agent model was shown ----------------------------------------------------
    def agent_calls(self) -> list[list[BaseMessage]]:
        """Message lists of calls made with the agent system prompt."""
        return [
            c
            for c in self.model.calls
            if c and str(c[0].content).startswith("You are the customer support agent")
        ]

    def agent_binds(self) -> list[dict[str, Any]]:
        """bind_tools() records of the agent (the router binds none)."""
        return self.model.bind_log


# --- embeddings ---------------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(fold(text))


class HashingEmbeddings(Embeddings):
    """Bag-of-words hashing: texts sharing words are close, unrelated texts are near-orthogonal."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class FakeSparse:
    """Offline stand-in for SparseEncoder."""

    model_name = "fake-sparse"

    @staticmethod
    def _vec(text: str) -> models.SparseVector:
        counts: dict[int, float] = {}
        for tok in _tokens(text):
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % 100_000
            counts[idx] = counts.get(idx, 0.0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def encode_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._vec(t) for t in texts]

    def encode_query(self, text: str) -> models.SparseVector:
        return self._vec(text)
