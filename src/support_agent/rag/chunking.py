"""Heading-aware chunking.

Each heading section becomes its own chunk so citations point at one precise section.
Sections longer than the token budget are split on paragraph boundaries with overlap;
tables are kept whole where they fit. Token counts are an approximation (no tokenizer
download needed) tuned to over- rather than under-estimate Vietnamese and English text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from support_agent.core.i18n import detect_language
from support_agent.core.settings import RetrievalConfig
from support_agent.rag.loaders import LoadedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_TOKEN_PIECE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    """~1.35 tokens per word/punctuation piece; non-ASCII pieces cost more (subword splits)."""
    total = 0.0
    for piece in _TOKEN_PIECE.findall(text):
        total += 1.6 if not piece.isascii() else 1.2
    return math.ceil(total)


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    source: str
    section: str
    text: str
    lang: str
    updated_at: str
    content_hash: str
    chunk_index: int

    @property
    def embed_text(self) -> str:
        """Text sent to the encoders: the section breadcrumb gives short chunks context."""
        return f"{self.doc_id} > {self.section}\n{self.text}"


@dataclass
class _Section:
    path: list[str]
    lines: list[str]


def _split_sections(text: str) -> list[_Section]:
    sections: list[_Section] = [_Section(path=[], lines=[])]
    stack: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(_Section(path=[t for _, t in stack], lines=[]))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if "".join(s.lines).strip()]


def _paragraphs(lines: list[str]) -> list[str]:
    """Blank-line separated blocks. A markdown table stays one block."""
    blocks: list[str] = []
    cur: list[str] = []
    for line in lines:
        if line.strip():
            cur.append(line)
        elif cur:
            blocks.append("\n".join(cur))
            cur = []
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _split_long_block(block: str, max_tokens: int) -> list[str]:
    """Split one over-budget block (long paragraph or table) by line, then by sentence."""
    units = block.split("\n") if "\n" in block else re.split(r"(?<=[.!?。])\s+", block)
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    sep = "\n" if "\n" in block else " "
    for unit in units:
        tok = estimate_tokens(unit)
        if cur and cur_tokens + tok > max_tokens:
            out.append(sep.join(cur))
            cur, cur_tokens = [], 0
        cur.append(unit)
        cur_tokens += tok
    if cur:
        out.append(sep.join(cur))
    return out


def _pack(blocks: list[str], cfg: RetrievalConfig) -> list[str]:
    max_t, min_t = cfg.chunk_tokens.max, cfg.chunk_tokens.min
    expanded: list[str] = []
    for b in blocks:
        expanded.extend(_split_long_block(b, max_t) if estimate_tokens(b) > max_t else [b])

    packed: list[list[str]] = []
    cur: list[str] = []
    cur_tokens = 0
    for b in expanded:
        tok = estimate_tokens(b)
        if cur and cur_tokens + tok > max_t:
            packed.append(cur)
            overlap = _overlap_tail(cur, int(max_t * cfg.chunk_overlap_ratio))
            cur = list(overlap)
            cur_tokens = sum(estimate_tokens(x) for x in cur)
        cur.append(b)
        cur_tokens += tok
    if cur:
        # Avoid a tiny trailing chunk: fold it into the previous one if it fits.
        if packed and cur_tokens < min_t // 2:
            merged = packed[-1] + [b for b in cur if b not in packed[-1]]
            if sum(estimate_tokens(x) for x in merged) <= int(max_t * 1.25):
                packed[-1] = merged
                cur = []
        if cur:
            packed.append(cur)
    return ["\n\n".join(p) for p in packed]


def _tail_slice(text: str, budget: int) -> str:
    """The trailing sentences (or, failing that, words) of `text` that fit in `budget` tokens."""
    for units, sep in ((re.split(r"(?<=[.!?。])\s+", text), " "), (text.split(), " ")):
        out: list[str] = []
        used = 0
        for unit in reversed(units):
            tok = estimate_tokens(unit)
            if used + tok > budget:
                break
            out.insert(0, unit)
            used += tok
        if out:
            return sep.join(out)
    return ""


def _overlap_tail(blocks: list[str], budget: int) -> list[str]:
    """Blocks to repeat at the start of the next chunk, within `budget` tokens.

    Whole trailing blocks are used when they fit; if even the last block is bigger than the
    budget, its tail (sentences, then words) is used so overlap still happens.
    """
    tail: list[str] = []
    used = 0
    for b in reversed(blocks):
        tok = estimate_tokens(b)
        if used + tok <= budget:
            tail.insert(0, b)
            used += tok
            continue
        if not tail:
            piece = _tail_slice(b, budget)
            if piece:
                tail.insert(0, piece)
        break
    return tail


def chunk_document(doc: LoadedDocument, cfg: RetrievalConfig) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in _split_sections(doc.text):
        # The section label is what users see in citations: keep only the deepest two levels.
        label = " > ".join(section.path[-2:]) if section.path else "Overview"
        for text in _pack(_paragraphs(section.lines), cfg):
            text = text.strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    section=label,
                    text=text,
                    lang=detect_language(text),
                    updated_at=doc.updated_at,
                    content_hash=doc.content_hash,
                    chunk_index=len(chunks),
                )
            )
    return chunks
