"""Input and output guardrails (SPEC 13.1, 13.5) and the per-user rate limiter.

These are cheap, deterministic checks that run on every turn. They are a safety net, not the
primary defence: identity is never an LLM argument, ownership is part of each query, and
untrusted text is wrapped as data. A heuristic can be evaded; the structure around it cannot.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from support_agent.core.settings import GuardrailsConfig
from support_agent.security.pii import EMAIL, PHONE, mask_email, mask_phone

# --- input ---------------------------------------------------------------------------

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏‪-‮⁦-⁩]")

# High-precision phrases only: an ordinary shopping question must never trip these.
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        (
            "override_instructions",
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|"
            r"all|your|system|these|the)\b[^.\n]{0,30}\b(instructions?|rules?|prompts?|guidelines?|"
            r"restrictions?|polic(y|ies))\b",
        ),
        (
            "forget_context",
            r"\b(forget|ignore|disregard)\b[^.\n]{0,15}\b(everything|all)\b[^.\n]{0,25}"
            r"\b(above|before|prior|previous|so far|you (were|have been) (told|given)|you know)\b",
        ),
        (
            "override_instructions_vi",
            r"\b(bỏ qua|phớt lờ|quên|vượt qua|bo qua|quen)\b[^.\n]{0,40}\b(hướng dẫn|chỉ dẫn|"
            r"quy tắc|luật|lệnh|huong dan|chi dan|quy tac)\b",
        ),
        (
            "reveal_prompt",
            r"\b(reveal|show|print|repeat|display|leak|tell me|give me|output)\b[^.\n]{0,40}"
            r"\b(system|hidden|initial|developer|original)\b[^.\n]{0,15}\b(prompt|instructions?|"
            r"message|configuration)\b",
        ),
        (
            "reveal_prompt_vi",
            r"\b(tiết lộ|hiển thị|cho (tôi )?xem|in ra|lặp lại|tiet lo|cho toi xem)\b[^.\n]{0,40}"
            r"\b(system prompt|prompt hệ thống|chỉ dẫn hệ thống|hướng dẫn ban đầu|"
            r"prompt he thong)\b",
        ),
        ("persona_switch", r"\b(you are now|from now on you are|pretend (to be|you are))\b"),
        (
            "persona_switch_vi",
            r"\b(bây giờ bạn là|từ giờ bạn là|hãy đóng vai|giả vờ (bạn )?là|bay gio ban la)\b",
        ),
        (
            "jailbreak",
            r"\b(jailbreak|developer mode|dan mode|do anything now|sudo mode|god mode)\b|"
            r"chế độ (nhà phát triển|không giới hạn)",
        ),
        (
            "role_claim",
            r"\b(i am|i'm|this is|tôi là|mình là)\s+(the\s+|a\s+|an\s+|một\s+)?"
            r"(admin(istrator)?|staff|employee|developer|nhân viên|quản trị( viên)?)\b",
        ),
        (
            "identity_override",
            r"\b(customer_id|user_id|principal)\s*[:=]|"
            r"\bact (as|on behalf of) (customer|user)\s+\w+|"
            r"\b(as|for) (customer|user) (id )?u_\d+",
        ),
        (
            "delimiter_forgery",
            r"</?\s*(untrusted_data|system|documents|customer_preferences)\b|<\|im_(start|end)\|>|"
            r"\[/?INST\]|^\s*(system|assistant)\s*:",
        ),
    )
)


@dataclass(frozen=True)
class InputVerdict:
    text: str  # normalised text (control characters removed)
    blocked: bool = False
    reasons: tuple[str, ...] = ()


def clean_input(text: str) -> str:
    """NFC-normalise and drop control / bidi characters that can hide instructions."""
    text = unicodedata.normalize("NFC", text)
    return _CONTROL.sub("", text).strip()


def check_input(text: str, cfg: GuardrailsConfig) -> InputVerdict:
    """Clean `text` and decide whether it may reach the model.

    Length is not enforced here: the agent truncates (it still answers) and the API rejects
    overlong bodies with a 400. This function only looks for injection attempts.
    """
    cleaned = clean_input(text)
    if not cfg.injection_detection:
        return InputVerdict(cleaned)
    reasons = tuple(name for name, rx in _INJECTION_PATTERNS if rx.search(cleaned))
    return InputVerdict(cleaned, blocked=bool(reasons), reasons=reasons)


# --- rate limit ----------------------------------------------------------------------


class RateLimiter:
    """Sliding one-minute window per key (SPEC 13.1). In-process: one limiter per API worker."""

    def __init__(
        self,
        per_minute: int,
        *,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = per_minute
        self.window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str) -> float:
        """Count a request. Returns 0 if allowed, else the seconds until a slot frees up."""
        if self.limit <= 0:
            return 0.0
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(0.001, self.window - (now - hits[0]))
        hits.append(now)
        if len(self._hits) > 10_000:  # forget idle keys so the table cannot grow without bound
            for stale in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                del self._hits[stale]
        return 0.0


# --- output --------------------------------------------------------------------------

_SECRETS = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}|AIza[0-9A-Za-z_-]{20,}|sk-ant-[A-Za-z0-9_-]{16,}|"
    r"lf_(?:sk|pk)_[A-Za-z0-9-]{8,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,})"
)

_NEGATION = re.compile(
    r"\b(not|n't|no|cannot|can't|unable|isn't|aren't|won't|never|unfortunately)\b|"
    r"\b(không|chưa|chẳng|khó|rất tiếc|tiếc)\b",
    re.IGNORECASE,
)
# A definitive promise that money or approval has been granted. The agent can only ever
# *propose* a request: staff decide (SPEC 9.4).
_PROMISES = re.compile(
    r"\b(has|have|had|was|were|been|is|are|will be|'ll be|is going to be)\b[^.\n]{0,20}\b"
    r"(refunded|approved|accepted|guaranteed)\b|"
    r"\b(we|i)\s*(will|'ll|shall)\s*(definitely\s*)?(refund|reimburse|approve)\b|"
    r"\b(đã|sẽ|được)\s*(được\s*)?(hoàn tiền|duyệt|chấp nhận|chấp thuận)\b|"
    r"\b(chắc chắn|đảm bảo)\b[^.\n]{0,30}\b(hoàn tiền|được duyệt)\b",
    re.IGNORECASE,
)
_SENTENCES = re.compile(r"(?<=[.!?\n])\s+")


@dataclass
class OutputVerdict:
    text: str
    changed: bool = False
    flags: list[str] = field(default_factory=list)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _shingles(text: str, size: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + size]) for i in range(max(0, len(words) - size + 1))}


def promises_outcome(text: str) -> bool:
    """True if some sentence states that a refund/approval is granted (and does not negate it)."""
    return any(
        _PROMISES.search(s) and not _NEGATION.search(s) for s in _SENTENCES.split(text) if s.strip()
    )


def check_output(
    text: str,
    *,
    cfg: GuardrailsConfig,
    known_text: Iterable[str] = (),
    system_prompt: str = "",
    ineligible: bool = False,
    safe_leak: str = "",
    safe_promise: str = "",
) -> OutputVerdict:
    """Make `text` safe to show.

    * `known_text`: what this turn legitimately saw (the caller's own tool results and message).
      An e-mail address or phone number that appears in none of it was not read from the
      caller's data, so it is masked (SPEC 13.5).
    * `system_prompt`: any long verbatim run from it is a leak.
    * `ineligible`: a rule returned `eligible=false` this turn, so any promise is a false one.
    """
    verdict = OutputVerdict(text)

    if _SECRETS.search(verdict.text) or (
        system_prompt and len(_shingles(system_prompt) & _shingles(verdict.text)) >= 1
    ):
        return OutputVerdict(safe_leak, True, ["prompt_or_secret_leak"])

    if ineligible and promises_outcome(verdict.text):
        # Drop only the sentences that promise, so the reason the model gave survives.
        kept = [
            s
            for s in _SENTENCES.split(verdict.text)
            if s.strip() and not (_PROMISES.search(s) and not _NEGATION.search(s))
        ]
        body = " ".join(s.strip() for s in kept)
        text = f"{body}\n\n{safe_promise}" if len(body) >= 20 and safe_promise else safe_promise
        return OutputVerdict(text, True, ["false_commitment"])

    if cfg.pii_masking:
        known = " ".join(known_text)
        known_digits = _digits(known)
        known_lower = known.lower()
        masked = verdict.text

        def keep_phone(match: re.Match[str]) -> str:
            digits = _digits(match.group(0))
            return (
                match.group(0)
                if digits and digits[-9:] in known_digits
                else mask_phone(match.group(0))
            )

        def keep_email(match: re.Match[str]) -> str:
            return (
                match.group(0)
                if match.group(0).lower() in known_lower
                else mask_email(match.group(0))
            )

        masked = EMAIL.sub(keep_email, masked)
        masked = PHONE.sub(keep_phone, masked)
        if masked != verdict.text:
            verdict.text, verdict.changed = masked, True
            verdict.flags.append("pii_masked")
    return verdict
