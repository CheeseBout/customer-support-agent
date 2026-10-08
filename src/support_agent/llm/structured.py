"""Structured output with a JSON-parsing fallback for models that lack native support."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def message_text(message: BaseMessage) -> str:
    """Flatten message content (str or list of content blocks) into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def extract_json(text: str) -> Any:
    """Parse the first JSON object found in `text`, tolerating code fences and prose."""
    fenced = _FENCE.search(text)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start : i + 1])
    raise ValueError("unterminated JSON object")


async def structured_invoke[T: BaseModel](
    model: BaseChatModel,
    schema: type[T],
    messages: list[BaseMessage],
    *,
    prefer_native: bool = True,
    config: RunnableConfig | None = None,
) -> T:
    """Return `schema` parsed from the model, using native structured output when possible.

    Falls back to prompting for JSON when native support is missing or fails (SPEC 6.2,
    "degraded" mode).
    """
    if prefer_native:
        try:
            result = await model.with_structured_output(schema).ainvoke(messages, config=config)
            if isinstance(result, schema):
                return result
            if isinstance(result, dict):
                return schema.model_validate(result)
        except (NotImplementedError, ValidationError, ValueError, TypeError) as exc:
            log.warning("native structured output failed, using JSON fallback: %s", exc)
        except Exception as exc:  # provider-specific API errors (400 on unsupported params)
            log.warning("native structured output errored, using JSON fallback: %s", exc)

    instruction = (
        "Respond with a single JSON object only, no prose, matching this JSON schema:\n"
        + json.dumps(schema.model_json_schema())
    )
    fallback_messages = [*messages, SystemMessage(content=instruction)]
    if not any(isinstance(m, HumanMessage) for m in fallback_messages):
        fallback_messages.append(HumanMessage(content="Respond now."))
    response = await model.ainvoke(fallback_messages, config=config)
    return schema.model_validate(extract_json(message_text(response)))
