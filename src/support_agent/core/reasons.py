"""Why a customer is returning or claiming on an item. Shared by the drafts and the shop's rules."""

from __future__ import annotations

from typing import Literal, get_args

ReasonCode = Literal[
    "defective", "wrong_item", "not_as_described", "damaged_in_transit", "changed_mind", "other"
]
REASON_CODES: tuple[str, ...] = get_args(ReasonCode)
