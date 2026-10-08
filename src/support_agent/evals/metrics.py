"""Pure metric functions. No I/O and no LLM, so they are cheap to test exhaustively."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from support_agent.core.i18n import VIETNAMESE_CHARS, detect_language
from support_agent.evals.dataset import Fact

# --- text matching -------------------------------------------------------------------

_THOUSANDS = re.compile(r"(?<=\d)[.,  ](?=\d{3}(?!\d))")


def fold_text(text: str) -> str:
    """Lowercase, strip diacritics and thousands separators (350.000 = 350,000 = 350000)."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    text = _THOUSANDS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def fact_present(answer: str, fact: Fact) -> bool:
    """`fact` is a string, or a list of alternatives where any one is enough."""
    haystack = fold_text(answer)
    alternatives = [fact] if isinstance(fact, str) else fact
    return any(fold_text(a) in haystack for a in alternatives)


def facts_satisfied(answer: str, facts: Sequence[Fact]) -> tuple[int, int]:
    """(facts found, facts expected)."""
    return sum(1 for f in facts if fact_present(answer, f)), len(facts)


def forbidden_hits(answer: str, must_not: Sequence[str]) -> list[str]:
    haystack = fold_text(answer)
    return [m for m in must_not if fold_text(m) in haystack]


def language_ok(answer: str, expected: str) -> bool:
    """Judge the language of an *answer* by the share of Vietnamese-looking words.

    `detect_language` treats any Vietnamese letter as decisive, which suits short user input but
    misreads an English answer that quotes a Vietnamese product name ("Tai nghe Bluetooth").
    """
    words = re.findall(r"\w+", answer)
    if not words:
        return True
    vi_share = sum(1 for w in words if any(c in VIETNAMESE_CHARS for c in w)) / len(words)
    if vi_share >= 0.2:
        return expected == "vi"
    if vi_share == 0 and detect_language(answer) == "vi":  # Vietnamese typed without accents
        return expected == "vi"
    return expected == "en"


_NEGATIVE = (
    "not eligible", "isn't eligible", "ineligible", "cannot", "can't", "unable", "no longer",
    "expired", "unfortunately", "not possible", "not able", "not allowed", "already",
    "khong du dieu kien", "khong the", "khong duoc", "da het han", "qua han", "rat tiec",
    "khong con", "da qua", "khong thuoc", "da co yeu cau",
)  # fmt: skip
_POSITIVE = (
    "eligible", "can return", "you can", "yes", "still within", "within the",
    "du dieu kien", "co the", "con duoc", "duoc doi tra", "con han", "con thoi han",
    "con bao hanh", "con hieu luc", "dang duoc bao hanh", "under warranty", "covered",
    "van con", "con trong thoi", "van duoc bao hanh",
)  # fmt: skip


def _polarity(text: str) -> str:
    folded = fold_text(text)
    if any(m in folded for m in _NEGATIVE):
        return "no"
    if any(m in folded for m in _POSITIVE):
        return "yes"
    return "unknown"


def eligibility_polarity(answer: str) -> str:
    """Rough read of whether an answer says "yes, eligible" or "no". Negations win.

    The opening sentence carries the verdict; later sentences list exclusions ("this does not
    cover water damage") that must not flip it, so they are read only when it is unclear.
    """
    opening = re.split(r"(?<=[.!?])\s|\n", answer.strip(), maxsplit=1)[0]
    return _polarity(opening) if _polarity(opening) != "unknown" else _polarity(answer)


def draft_ok(
    *,
    expected_type: str | None,
    amount: int | None,
    priority: bool | None,
    contains: Sequence[str],
    decision: str | None,
    interrupt: dict[str, Any] | None,
    created: Sequence[dict[str, Any]],
) -> bool:
    """Did the request flow end where the dataset says it should?

    `interrupt` is the last confirmation shown to the customer; `created` the drafts that
    exist afterwards. Everything is read from what our own code produced, never from the
    model's wording.
    """
    if expected_type is None:
        return interrupt is None and not created  # nothing proposed, nothing stored
    if interrupt is None or interrupt.get("draft_type") != expected_type:
        return False
    summary = interrupt.get("summary") or {}
    shown = summary.get("amount", summary.get("total"))
    if amount is not None and shown != amount:
        return False
    if priority is not None and bool(interrupt.get("priority_review")) != priority:
        return False
    text = fold_text(json.dumps(summary, ensure_ascii=False))
    if any(fold_text(part) not in text for part in contains):
        return False
    if decision == "approve":
        return len(created) == 1 and created[0].get("type") == expected_type
    return not created  # reject, edit and "no answer" must not store anything


POLICY_SEARCH = "search_policy"


NOT_DATA_TOOLS = frozenset({POLICY_SEARCH, "load_skill"})


def data_tools(called: Sequence[str]) -> list[str]:
    """The tools that touch the customer's data. Policy search is retrieval, judged separately:
    the Phase A pipeline ran it implicitly while the agent calls it as a tool, and the same
    question must score the same under both. Loading a skill reads a procedure, not data."""
    return [t for t in called if t not in NOT_DATA_TOOLS]


def tools_ok(expected: Sequence[str], called: Sequence[str]) -> bool:
    """Every expected tool was called; and a question that needs no tools called none."""
    if not expected:
        return not called
    return set(expected) <= set(called)


# --- retrieval ----------------------------------------------------------------------


def first_rank(
    items: Sequence[tuple[str, str]], docs: Sequence[str], sections: Sequence[str]
) -> tuple[int | None, int | None]:
    """1-based rank of the first retrieved (doc_id, section) matching a doc / a section hint."""
    doc_rank = section_rank = None
    for i, (doc_id, section) in enumerate(items, start=1):
        if doc_rank is None and doc_id in docs:
            doc_rank = i
        if section_rank is None and doc_id in docs and any(s in section for s in sections):
            section_rank = i
    return doc_rank, section_rank


def hit_at_k(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def mean(values: Sequence[float | bool]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class GatePoint:
    threshold: float
    recall: float  # share of answerable questions whose best chunk passes the gate
    specificity: float  # share of unanswerable questions correctly rejected

    @property
    def balanced(self) -> float:
        return (self.recall + self.specificity) / 2


def gate_curve(
    positive_scores: Sequence[float], negative_scores: Sequence[float], thresholds: Sequence[float]
) -> list[GatePoint]:
    def share(scores: Sequence[float], pred: Callable[[float], bool]) -> float:
        return sum(1 for s in scores if pred(s)) / len(scores) if scores else 1.0

    return [
        GatePoint(
            threshold=t,
            recall=share(positive_scores, lambda s, t=t: s >= t),  # type: ignore[misc]
            specificity=share(negative_scores, lambda s, t=t: s < t),  # type: ignore[misc]
        )
        for t in thresholds
    ]


def recommend_threshold(
    positive_scores: Sequence[float], negative_scores: Sequence[float], curve: Sequence[GatePoint]
) -> float | None:
    """The midpoint of the gap when the classes separate cleanly, else the best balanced point."""
    if not positive_scores or not negative_scores:
        return None
    lowest_pos, highest_neg = min(positive_scores), max(negative_scores)
    if highest_neg < lowest_pos:
        return round((lowest_pos + highest_neg) / 2, 3)
    best = max(curve, key=lambda p: (p.balanced, -p.threshold))
    return best.threshold


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]
