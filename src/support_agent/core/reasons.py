"""Why a customer is returning or claiming on an item. Shared by the drafts and the shop's rules."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_args

ReasonCode = Literal[
    "defective", "wrong_item", "not_as_described", "damaged_in_transit", "changed_mind", "other"
]
REASON_CODES: tuple[str, ...] = get_args(ReasonCode)

# What a customer says when they give a reason, in accent-folded lower case (Vietnamese and
# English). Phrases are matched as whole words. They are deliberately specific: "loi" alone would
# match "xin loi" (sorry), and "hong" alone would match the colour pink (hồng).
_REASON_PHRASES: dict[str, tuple[str, ...]] = {
    "defective": (
        "defective", "faulty", "broken", "malfunction", "stopped working", "not working",
        "is dead", "went dead", "dead on arrival",
        "does not work", "doesn't work", "doesnt work", "dead pixel", "flicker", "flickering",
        "bi loi", "hang loi", "san pham loi", "co loi", "loi ky thuat", "loi nha san xuat",
        "bi hong", "hu hong", "khong hoat dong", "khong dung duoc", "khong len nguon",
        "chap chon", "nhap nhay", "so chi", "bung chi", "tuot chi",
    ),
    "damaged_in_transit": (
        "damaged in transit", "arrived damaged", "damaged during shipping", "damaged on arrival",
        "hu hong khi van chuyen", "mop meo", "bi dap khi giao", "vo khi van chuyen",
    ),
    "wrong_item": (
        "wrong item", "wrong product", "sent the wrong", "not what i ordered",
        "giao sai", "sai hang", "nham hang", "giao nham", "nham size", "sai size",
    ),
    "not_as_described": (
        "not as described", "does not match the description", "different from the description",
        "khong dung mo ta", "khac voi mo ta", "khong giong hinh", "khong giong mo ta",
    ),
    "changed_mind": (
        "changed my mind", "change of mind", "do not like", "don't like", "dont like",
        "does not fit", "doesn't fit", "doesnt fit", "too small", "too big", "too large",
        "too tight", "too loose", "khong vua", "doi y", "khong thich", "chat qua", "rong qua",
        "nho qua", "to qua",
    ),
}  # fmt: skip
# When a message matches several, the first of these wins: the specific fault reasons go first,
# because shops give those the longest period and they must not be hidden behind a milder one.
_PRIORITY = (
    "damaged_in_transit",  # the more specific phrase wins over a plain fault
    "defective",
    "wrong_item",
    "not_as_described",
    "changed_mind",
)


def fold_words(text: str) -> str:
    """Lower case without accents (đ -> d), single-spaced: the form `reasons_in` matches on."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"\s+", " ", text).strip()


def _matches(folded: str, phrases: Sequence[str]) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(p)}(?!\w)", folded) for p in phrases)


def reasons_in(text: str, extra: Mapping[Any, Sequence[str]] | None = None) -> list[str]:
    """The reasons a customer's message already gives, most important first.

    `extra` adds a shop's own phrases per reason code (`router.reason_words`); they are folded
    the same way as the message.
    """
    folded = fold_words(text)
    found = []
    for code in _PRIORITY:
        phrases = list(_REASON_PHRASES.get(code, ()))
        phrases += [fold_words(p) for p in (extra or {}).get(code, ())]
        if _matches(folded, phrases):
            found.append(code)
    return found
