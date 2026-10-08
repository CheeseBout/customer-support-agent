"""Regex extraction of order ids and SKUs (SPEC FR-007). Exact lookup, never embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from support_agent.core.settings import RouterConfig

_FLAGS = re.IGNORECASE

# group(1) is always the id.
_BUILTIN_ORDER_PATTERNS = [
    # "#1234", "#ORD-1234"
    r"#\s*([A-Za-z0-9][A-Za-z0-9-]{2,19})",
    # "order 1234", "order id: 1234", "đơn hàng số 1234", "mã đơn 1234", "don hang 1234"
    r"(?:\border|\bđơn(?:\s+hàng)?|\bdon(?:\s+hang)?|\bmã\s+đơn(?:\s+hàng)?|\bma\s+don(?:\s+hang)?)"
    r"(?:\s+(?:id|number|no\.?|số|so|mã|ma))?\s*[:#]?\s*([A-Za-z]{0,5}-?\d{3,12})\b",
    # bare prefixed codes: ORD-1234, DH1234
    r"\b((?:ORD|DH|OD)-?\d{3,12})\b",
]
_BUILTIN_SKU_PATTERN = r"\b([A-Z]{2,6}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*)\b"
_ORDER_LIKE = re.compile(r"^(?:ORD|DH|OD)-?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class Entities:
    order_ids: list[str] = field(default_factory=list)
    skus: list[str] = field(default_factory=list)

    def merge(self, other: Entities) -> Entities:
        return Entities(
            order_ids=_dedupe([*self.order_ids, *other.order_ids]),
            skus=_dedupe([*self.skus, *other.skus]),
        )

    @property
    def empty(self) -> bool:
        return not self.order_ids and not self.skus


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalise_order_id(raw: str) -> str:
    return raw.strip().lstrip("#").strip().upper()


def extract_entities(text: str, cfg: RouterConfig | None = None) -> Entities:
    cfg = cfg or RouterConfig()
    order_ids: list[str] = []
    for pattern in [*_BUILTIN_ORDER_PATTERNS, *cfg.order_id_patterns]:
        for m in re.finditer(pattern, text, _FLAGS):
            candidate = normalise_order_id(m.group(1))
            # A '#' reference must contain a digit to be an order id ("#help" is not one).
            if candidate and any(ch.isdigit() for ch in candidate):
                order_ids.append(candidate)
    order_ids = _dedupe(order_ids)

    sku_re = re.compile(cfg.sku_pattern or _BUILTIN_SKU_PATTERN)  # case-sensitive by design
    skus = [
        m.group(1)
        for m in sku_re.finditer(text)
        if m.group(1).upper() not in order_ids and not _ORDER_LIKE.match(m.group(1))
    ]
    return Entities(order_ids=order_ids, skus=_dedupe(skus))
