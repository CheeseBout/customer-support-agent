"""Policy retriever: hybrid search, cosine gate, optional rerank."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from support_agent.core.settings import RetrievalConfig
from support_agent.rag.embeddings import cosine
from support_agent.rag.index import VectorStore
from support_agent.rag.sparse import SparseEncoder


class RetrievedChunk(BaseModel):
    text: str
    doc_id: str
    source: str
    section: str
    lang: str
    updated_at: str
    score: float  # cosine similarity of the dense vectors (the gated quantity)
    rank: int


def normalise_query(query: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", query)).strip()


class Reranker:
    """Optional cross-encoder (multilingual) applied after the cosine gate."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        cache_dir: Path | str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model: Any = None

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(self.model_name, cache_dir=self.cache_dir)
        return [float(s) for s in self._model.rerank(query, texts)]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: Embeddings,
        cfg: RetrievalConfig,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.cfg = cfg
        self.sparse = sparse if cfg.hybrid else None
        self.reranker = reranker if cfg.rerank else None

    def retrieve_sync(
        self, query: str, *, top_k: int | None = None, threshold: float | None = None
    ) -> list[RetrievedChunk]:
        """Empty list means "no sufficiently relevant policy text" (SPEC FR-005)."""
        query = normalise_query(query)
        if not query:
            return []
        top_k = top_k or self.cfg.top_k
        threshold = self.cfg.score_threshold if threshold is None else threshold

        dense_q = self.embeddings.embed_query(query)
        sparse_q = self.sparse.encode_query(query) if self.sparse else None
        fetch = top_k * 3 if self.reranker else top_k
        hits = self.store.search(dense_q, sparse_q, limit=fetch)

        scored: list[tuple[float, dict]] = []
        for hit in hits:
            if hit.dense_vector is not None:
                scored.append((cosine(dense_q, hit.dense_vector), hit.payload))

        # Gate on the query: is anything in the knowledge base close enough? Then keep the
        # hybrid (dense + keyword) ranking, dropping only clearly weak hits.
        if not scored or max(score for score, _ in scored) < threshold:
            return []
        floor = threshold - self.cfg.hit_floor_margin
        scored = [item for item in scored if item[0] >= floor]

        if self.reranker and len(scored) > 1:
            order = self.reranker.rerank(query, [p["text"] for _, p in scored])
            ranked = sorted(zip(order, scored, strict=True), key=lambda x: x[0], reverse=True)
            scored = [item for _, item in ranked]

        return [
            RetrievedChunk(
                text=p["text"],
                doc_id=p["doc_id"],
                source=p["source"],
                section=p["section"],
                lang=p["lang"],
                updated_at=p["updated_at"],
                score=round(score, 4),
                rank=i + 1,
            )
            for i, (score, p) in enumerate(scored[:top_k])
        ]

    async def retrieve(
        self, query: str, *, top_k: int | None = None, threshold: float | None = None
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.retrieve_sync, query, top_k=top_k, threshold=threshold)
