"""Streaming-safe handling of the two markers the model writes into its answer.

* `[NO_INFO]` as the very first thing: "the gathered material does not answer this".
* `[D1]`, `[D1, D2]`: citations of policy documents.

Neither belongs in what the customer reads, but the answer is streamed token by token, so a
marker can arrive split across chunks. `AnswerStream` holds back only the few characters that
might still turn into a marker and releases everything else immediately.
"""

from __future__ import annotations

import re

NO_INFO = "[NO_INFO]"

_CITATION = re.compile(r"[ \t]?\[D\d+(?:\s*,\s*D\d+)*\]")
# The tail of a buffer that could still grow into a citation: a trailing space (the marker
# swallows the space before it), "[", "[D", "[D1", "[D1,", "[D1, D" and the same after a space.
_PARTIAL_CITATION = re.compile(r"[ \t]?(?:\[(?:D[\dD,\s]*)?)?$")
_DOC_ID = re.compile(r"D(\d+)")


class AnswerStream:
    def __init__(self) -> None:
        self.raw = ""
        self.no_info = False
        self.cited: list[str] = []  # in order of first appearance, e.g. ["D1", "D3"]
        self._pending = ""
        self._checked_sentinel = False
        self._at_start = True

    def feed(self, chunk: str) -> str:
        """Text that is safe to show now (markers removed)."""
        self.raw += chunk
        buf = self._pending + chunk
        self._pending = ""

        if not self._checked_sentinel:
            probe = buf.lstrip()
            if not probe or (NO_INFO.startswith(probe) and len(probe) < len(NO_INFO)):
                self._pending = buf  # only whitespace so far, or could still become the sentinel
                return ""
            if probe.startswith(NO_INFO):
                self.no_info = True
                buf = probe[len(NO_INFO) :].lstrip()
            self._checked_sentinel = True

        hold = _PARTIAL_CITATION.search(buf)
        if hold and hold.group(0):
            self._pending, buf = buf[hold.start() :], buf[: hold.start()]
        return self._visible(self._strip_citations(buf))

    def _visible(self, text: str) -> str:
        """Drop whitespace before the first word, however the stream happened to be cut."""
        if self._at_start:
            text = text.lstrip()
            self._at_start = not text
        return text

    def finish(self) -> str:
        """Flush whatever was held back (an unfinished marker is just text)."""
        tail, self._pending = self._pending, ""
        if not self._checked_sentinel:
            self._checked_sentinel = True
            if tail.lstrip().startswith(NO_INFO):  # cannot happen mid-stream; kept for safety
                self.no_info = True
                tail = tail.lstrip()[len(NO_INFO) :].lstrip()
        return self._visible(self._strip_citations(tail))

    def _strip_citations(self, text: str) -> str:
        def take(match: re.Match[str]) -> str:
            for number in _DOC_ID.findall(match.group(0)):
                doc_id = f"D{number}"
                if doc_id not in self.cited:
                    self.cited.append(doc_id)
            return ""

        return _CITATION.sub(take, text)
