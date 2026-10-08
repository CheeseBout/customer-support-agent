"""PII masking for logs and answers (SPEC NFR-011) and removal of secrets before storage."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9-]+\.[A-Za-z0-9.-]+)\b")
# VN mobile (0xxxxxxxxx / +84xxxxxxxxx) with optional separators.
_PHONE = re.compile(r"(?<![\d*])(?:\+84|84|0)[ .-]?\d(?:[ .-]?\d){7,9}(?!\d)")
# Card-like long digit runs.
_LONG_DIGITS = re.compile(r"(?<!\d)\d{12,19}(?!\d)")

EMAIL = _EMAIL  # public handles for callers that need to match, not just mask
PHONE = _PHONE


def mask_email(text: str) -> str:
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def mask_phone(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        return "*" * (len(digits) - 3) + digits[-3:]

    return _PHONE.sub(repl, text)


def mask_text(text: str) -> str:
    text = mask_email(text)
    text = _LONG_DIGITS.sub(lambda m: "*" * (len(m.group(0)) - 4) + m.group(0)[-4:], text)
    return mask_phone(text)


# --- secrets that must never be stored or summarised (SPEC 12.1) -----------------------------

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PASSWORD = re.compile(
    r"(?i)(\b(?:password|passcode|passwd|cvv|cvc|mật khẩu|mat khau|mã pin|ma pin)\b"
    r"[ \t]*(?:is|la|là|:|=)?[ \t]*)\S+"
)
# "pin" alone means battery in Vietnamese ("pin tụt nhanh"): it is a secret only when it is
# followed by something that looks like a code (digits, optionally after "is" / ":" / "=").
_PIN_CODE = re.compile(r"(?i)(\bpin\b[ \t]*(?:is|la|là|:|=)?[ \t]*)(?=\S*\d)\S+")
REDACTED = "[redacted]"


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


def redact_secrets(text: str) -> str:
    """Remove card numbers (Luhn-valid only, so order codes survive) and stated passwords/PINs.

    A request id is a UUID: a run of digits inside one (the all-zero id used in tests, or a real
    one with few letters) is not a card number and must reach the tool unchanged.
    """
    ids = [m.span() for m in _UUID.finditer(text)]

    def card(m: re.Match[str]) -> str:
        inside = any(m.start() < end and m.end() > start for start, end in ids)
        return REDACTED if _luhn_ok(m.group(0)) and not inside else m.group(0)

    text = _CARD.sub(card, text)
    text = _PASSWORD.sub(lambda m: m.group(1) + REDACTED, text)
    return _PIN_CODE.sub(lambda m: m.group(1) + REDACTED, text)
