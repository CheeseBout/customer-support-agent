"""Incremental ingest: only re-process documents whose content (or encoder config) changed."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings

from support_agent.core.settings import KnowledgeConfig, RetrievalConfig
from support_agent.rag.audience import excluded, is_internal
from support_agent.rag.chunking import chunk_document
from support_agent.rag.index import VectorStore
from support_agent.rag.loaders import discover, file_hash, load_document
from support_agent.rag.sparse import SparseEncoder

log = logging.getLogger(__name__)

# Bump when the payload layout or chunking algorithm changes in a way that needs re-indexing.
INGEST_VERSION = "1"


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # not for customers: document -> why
    chunks_written: int = 0

    def summary(self) -> str:
        text = (
            f"added={len(self.added)} updated={len(self.updated)} unchanged={len(self.unchanged)} "
            f"removed={len(self.removed)} failed={len(self.failed)} chunks={self.chunks_written}"
        )
        return f"{text} skipped={len(self.skipped)}" if self.skipped else text


def make_ingest_key(
    content_hash: str, embedding_model: str, cfg: RetrievalConfig, hybrid: bool
) -> str:
    config = json.dumps(
        {
            "model": embedding_model,
            "chunk": cfg.chunk_tokens.model_dump(),
            "overlap": cfg.chunk_overlap_ratio,
            "hybrid": hybrid,
            "sparse": cfg.sparse_model,
            "v": INGEST_VERSION,
        },
        sort_keys=True,
    )
    return hashlib.sha256(f"{content_hash}|{config}".encode()).hexdigest()


def ingest(
    root: Path,
    *,
    store: VectorStore,
    embeddings: Embeddings,
    sparse: SparseEncoder | None,
    embedding_model: str,
    cfg: RetrievalConfig,
    full: bool = False,
    knowledge: KnowledgeConfig | None = None,
) -> IngestReport:
    """Sync `root` into the vector store. `full=True` rebuilds the collection from scratch.

    Documents that are not for customers (see `rag.audience`) are skipped, and removed from the
    index if an earlier run had indexed them."""
    report = IngestReport()
    know = knowledge or KnowledgeConfig()
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {root}")

    dim = len(embeddings.embed_query("dimension probe"))
    store.ensure_collection(dim, recreate=full)
    indexed = {} if full else store.indexed_documents()

    seen: set[str] = set()
    for path in discover(root):
        rel = path.relative_to(root).as_posix()
        if excluded(rel, know.exclude):
            report.skipped[rel] = "excluded by knowledge.exclude"
            continue
        try:
            doc = None
            if know.skip_internal or know.internal_markers:
                doc = load_document(path, root)
                if is_internal(doc.text, know.internal_markers):
                    report.skipped[rel] = "marked as internal (not for customers)"
                    log.warning("skipped %s: marked as internal, not for customers", rel)
                    continue
            seen.add(rel)
            key = make_ingest_key(file_hash(path), embedding_model, cfg, sparse is not None)
            if indexed.get(rel) == key:
                report.unchanged.append(rel)
                continue
            doc = doc or load_document(path, root)
            chunks = chunk_document(doc, cfg)
            if not chunks:
                raise ValueError("document produced no text chunks")
            texts = [c.embed_text for c in chunks]
            dense = embeddings.embed_documents(texts)
            sparse_vecs = sparse.encode_documents(texts) if sparse else None
            # Replace atomically-ish: drop stale chunks first so shrinking docs leave no orphans.
            if rel in indexed:
                store.delete_document(rel)
            store.upsert_chunks(
                chunks, dense, sparse_vecs, ingest_key=key, embedding_model=embedding_model
            )
            report.chunks_written += len(chunks)
            (report.updated if rel in indexed else report.added).append(rel)
            log.info("ingested %s (%d chunks)", rel, len(chunks))
        except Exception as exc:  # one bad file must not abort the whole run
            seen.add(rel)
            log.exception("failed to ingest %s", rel)
            report.failed[rel] = f"{type(exc).__name__}: {exc}"

    for doc_id in sorted(set(indexed) - seen):
        store.delete_document(doc_id)
        report.removed.append(doc_id)
    return report
