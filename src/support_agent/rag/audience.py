"""Who a policy document is for. Staff-only documents must not reach the customer-facing index.

Shops often keep internal procedures (complaint handling, staff regulations, approval limits) next
to their public policies. The assistant answers customers from everything it has indexed, so such a
document is a leak waiting for the right question, however well the router refuses it today.

A document is left out when its first lines say it is internal ("Audience: Internal", "Đối tượng:
Nội bộ", "internal use only", ...), when its path matches `knowledge.exclude`, or when it matches
one of the shop's own `knowledge.internal_markers`.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from fnmatch import fnmatch

from support_agent.core.reasons import fold_words

# Only the top of a document says who it is for; a policy that mentions "staff only" deep in its
# body (a store room, a rota) is not an internal document.
HEAD_CHARS = 3000

_INTERNAL = re.compile(
    r"(?:doi tuong|audience)\W{0,8}(?:noi bo|internal|staff only|nhan vien)"
    r"|chi dung noi bo|internal use only|for internal use|staff only|confidential"
)


def is_internal(text: str, extra_markers: Sequence[str] = ()) -> bool:
    """True when the start of `text` marks the document as staff-only."""
    head = fold_words(text[:HEAD_CHARS])
    if _INTERNAL.search(head):
        return True
    return any(re.search(fold_words(m), head) for m in extra_markers)


def excluded(rel_path: str, patterns: Sequence[str]) -> bool:
    """True when `rel_path` (posix, relative to the knowledge dir) matches any glob pattern."""
    return any(fnmatch(rel_path, p) or fnmatch(rel_path.rsplit("/", 1)[-1], p) for p in patterns)
