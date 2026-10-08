"""Language detection (VI/EN) and the user-facing message catalog."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

Lang = Literal["vi", "en"]

# Letters that exist in Vietnamese but not in English (after NFC normalisation).
_VI_CHARS = set(
    "ăâđêôơưĂÂĐÊÔƠƯ"
    "àáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"
    "ÀÁẠẢÃẦẤẬẨẪẰẮẶẲẴÈÉẸẺẼỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕỒỐỘỔỖỜỚỢỞỠÙÚỤỦŨỪỨỰỬỮỲÝỴỶỸ"
)

VIETNAMESE_CHARS = _VI_CHARS  # public alias for callers that need the character set

# Common Vietnamese words typed *without* diacritics ("don hang cua toi o dau").
_VI_UNACCENTED = {
    "toi",
    "tui",
    "ban",
    "khong",
    "duoc",
    "cua",
    "don",
    "hang",
    "bao",
    "nhieu",
    "ngay",
    "doi",
    "tra",
    "hoan",
    "tien",
    "giao",
    "san",
    "pham",
    "con",
    "het",
    "nao",
    "dau",
    "the",
    "gi",
    "cho",
    "minh",
    "xin",
    "chao",
    "cam",
    "on",
    "hanh",
    "ma",
    "va",
    "hay",
    "nhu",
    "sao",
    "thanh",
    "toan",
    "van",
    "chuyen",
    "mua",
    "muon",
    "lam",
}
_EN_COMMON = {
    "the",
    "is",
    "are",
    "my",
    "i",
    "do",
    "how",
    "what",
    "where",
    "when",
    "can",
    "order",
    "return",
    "refund",
    "days",
    "item",
    "of",
    "to",
    "a",
    "an",
    "and",
    "have",
    "still",
    "ship",
    "shipping",
    "warranty",
    "stock",
    "please",
    "you",
    "your",
    "me",
    "it",
    "in",
}
_WORD = re.compile(r"[a-zA-ZÀ-ỹ]+")


def detect_language(text: str, default: Lang = "en") -> Lang:
    """Heuristic detector. Diacritics decide; otherwise compare common-word hits."""
    text = unicodedata.normalize("NFC", text)
    if any(ch in _VI_CHARS for ch in text):
        return "vi"
    words = [w.lower() for w in _WORD.findall(text)]
    if not words:
        return default
    vi_hits = sum(1 for w in words if w in _VI_UNACCENTED and w not in _EN_COMMON)
    en_hits = sum(1 for w in words if w in _EN_COMMON)
    if vi_hits >= 2 and vi_hits > en_hits:
        return "vi"
    return "en" if en_hits or not vi_hits else default


def resolve_language(requested: str | None, text: str) -> Lang:
    """`requested` is `vi`, `en` or `auto`/None (detect from the text)."""
    if requested in ("vi", "en"):
        return requested  # type: ignore[return-value]
    return detect_language(text)


_MESSAGES: dict[str, dict[Lang, str]] = {
    "no_info": {
        "vi": "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong tài liệu chính sách hiện có.",
        "en": "Sorry, I couldn't find relevant information in the current policy documents.",
    },
    "order_not_found": {
        "vi": "Tôi không tìm thấy đơn hàng {order_id} trong tài khoản của bạn. Vui lòng kiểm tra lại mã đơn.",
        "en": "I couldn't find order {order_id} in your account. Please double-check the order ID.",
    },
    "clarify": {
        "vi": "Bạn có thể nói rõ hơn yêu cầu của mình không? Ví dụ: hỏi về chính sách, hoặc tra cứu một đơn hàng cụ thể.",
        "en": "Could you clarify your request? For example, a policy question or a lookup of a specific order.",
    },
    "out_of_scope": {
        "vi": "Tôi chỉ hỗ trợ các vấn đề về mua sắm: chính sách, đơn hàng, vận chuyển, sản phẩm và bảo hành.",
        "en": "I can only help with shopping topics: policies, orders, shipping, products and warranty.",
    },
    "chitchat": {
        "vi": "Xin chào! Tôi có thể giúp gì cho bạn về đơn hàng, chính sách hoặc sản phẩm?",
        "en": "Hello! How can I help you with orders, policies or products?",
    },
    "upstream_error": {
        "vi": "Hệ thống đang gặp sự cố tạm thời. Vui lòng thử lại sau ít phút.",
        "en": "We're having a temporary problem. Please try again in a few minutes.",
    },
    "step_limit": {
        "vi": "Tôi chưa thể hoàn tất yêu cầu này trong số bước cho phép. Bạn thử diễn đạt ngắn gọn hơn hoặc chia nhỏ câu hỏi nhé.",
        "en": "I couldn't finish this within the allowed number of steps. Please try rephrasing it more briefly or splitting it up.",
    },
    "stale_confirmation": {
        "vi": "Không còn yêu cầu nào đang chờ bạn xác nhận (có thể đã hết hạn hoặc đã được xử lý).",
        "en": "There is no request waiting for your confirmation any more (it may have expired or been handled).",
    },
    "confirmation_abandoned": {
        "vi": "Yêu cầu trước đó chưa được xác nhận nên đã bị huỷ; chưa có gì được gửi đi.",
        "en": "The earlier request was never confirmed, so it was dropped. Nothing was submitted.",
    },
    "timeout": {
        "vi": "Yêu cầu mất quá nhiều thời gian. Vui lòng thử lại hoặc diễn đạt ngắn gọn hơn.",
        "en": "The request took too long. Please try again or rephrase it more briefly.",
    },
    "confirm_prompt": {
        "vi": "Vui lòng xác nhận yêu cầu bên dưới trước khi tôi gửi đi.",
        "en": "Please confirm the request below before I submit it.",
    },
    "guardrail_blocked": {
        "vi": "Tôi không thể làm theo yêu cầu này. Tôi có thể giúp bạn về đơn hàng, chính sách, sản phẩm và bảo hành.",
        "en": "I can't follow that request. I can help with orders, policies, products and warranty.",
    },
    "rate_limited": {
        "vi": "Bạn gửi quá nhiều yêu cầu. Vui lòng thử lại sau {seconds} giây.",
        "en": "You are sending requests too quickly. Please try again in {seconds} seconds.",
    },
    "internal_hidden": {
        "vi": "Tôi không thể chia sẻ thông tin cấu hình nội bộ. Bạn cần hỗ trợ gì về đơn hàng hoặc chính sách?",
        "en": "I can't share internal configuration. What can I help you with about orders or policies?",
    },
    "no_promise": {
        "vi": "Theo kiểm tra, yêu cầu này hiện không đủ điều kiện nên tôi không thể cam kết hoàn tiền hay duyệt. Bạn có thể hỏi tôi lý do cụ thể hoặc cách khác để được hỗ trợ.",
        "en": "Based on the check, this request is not eligible, so I can't promise a refund or approval. You can ask me for the specific reason or for other ways to get help.",
    },
    "sources": {"vi": "Nguồn", "en": "Sources"},
}


def t(key: str, lang: Lang = "en", **fmt: object) -> str:
    entry = _MESSAGES[key]
    return entry.get(lang, entry["en"]).format(**fmt)
