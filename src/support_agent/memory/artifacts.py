"""Markdown files the agent leaves in a session's workspace (SPEC 12.3)."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_PAYLOAD = re.compile(r">\s*(\{.*\})\s*</", re.DOTALL)

_TITLES = {
    "comparison": {"vi": "So sánh sản phẩm", "en": "Product comparison"},
    "request": {"vi": "Tóm tắt yêu cầu", "en": "Request summary"},
}
_STATUS_LINE = {
    "vi": "Trạng thái: **{status}** (chờ nhân viên duyệt)",
    "en": "Status: **{status}** (waiting for staff review)",
}


def tool_data(message_text: str) -> dict[str, Any] | None:
    """The `data` of a wrapped tool result (`<untrusted_data ...>{json}</untrusted_data>`)."""
    found = _TOOL_PAYLOAD.search(message_text)
    if not found:
        return None
    try:
        parsed = json.loads(found.group(1))
    except ValueError:
        return None
    data = parsed.get("data") if isinstance(parsed, dict) else None
    return data if isinstance(data, dict) else None


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int | float):
        return f"{value:,}".replace(",", ".") if isinstance(value, int) else str(value)
    return str(value).replace("|", "/").replace("\n", " ")


def comparison_markdown(data: dict[str, Any], lang: str = "en") -> str | None:
    """A table from `compare_products` output, or None if it has no usable rows."""
    skus: list[str] = [str(s) for s in data.get("skus", [])]
    rows = [r for r in data.get("rows", []) if isinstance(r, dict)]
    if len(skus) < 2 or not rows:
        return None
    names: dict[str, Any] = next((r["values"] for r in rows if r.get("attribute") == "name"), {})
    header = ["", *(f"{names.get(s) or s} ({s})" for s in skus)]
    lines = [
        f"# {_TITLES['comparison'].get(lang, _TITLES['comparison']['en'])}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for row in rows:
        if row.get("attribute") == "name":
            continue
        values = row.get("values") or {}
        lines.append(
            "| "
            + " | ".join([str(row.get("attribute")), *(_cell(values.get(s)) for s in skus)])
            + " |"
        )
    return "\n".join(lines) + "\n"


def request_markdown(draft: dict[str, Any], summary: dict[str, Any], lang: str = "en") -> str:
    """What the customer asked for and where it stands, from the confirmed proposal."""
    title = _TITLES["request"].get(lang, _TITLES["request"]["en"])
    lines = [
        f"# {title}",
        "",
        f"- ID: `{draft['id']}`",
        f"- Type: {draft['type']}",
        "- " + _STATUS_LINE.get(lang, _STATUS_LINE["en"]).format(status=draft["status"]),
    ]
    if draft.get("priority_review"):
        lines.append("- Priority review: yes")
    lines.append("")
    for key, value in summary.items():
        if isinstance(value, list):
            lines.append(f"- {key}:")
            lines += [
                f"  - {_cell(item) if not isinstance(item, dict) else _item(item)}"
                for item in value
            ]
        else:
            lines.append(f"- {key}: {_cell(value)}")
    return "\n".join(lines) + "\n"


def _item(item: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {_cell(v)}" for k, v in item.items())
