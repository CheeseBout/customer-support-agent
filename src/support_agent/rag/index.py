"""Qdrant-backed vector store: dense + sparse (hybrid) over policy chunks."""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from support_agent.core.settings import Settings
from support_agent.rag.chunking import Chunk

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("6f1b3a52-3b7a-4c43-9d55-0c2f0d8f1a11")


@dataclass(frozen=True)
class Hit:
    payload: dict[str, Any]
    dense_vector: list[float] | None
    fusion_score: float


def make_client(settings: Settings) -> QdrantClient:
    """Server when QDRANT_URL is set, otherwise embedded storage at QDRANT_PATH."""
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    if settings.qdrant_path is None:
        raise ValueError("Set QDRANT_URL or QDRANT_PATH")
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}#{chunk_index}"))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    def close(self) -> None:
        self.client.close()

    # --- collection management -------------------------------------------------------
    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def dense_size(self) -> int | None:
        if not self.exists():
            return None
        cfg = self.client.get_collection(self.collection).config.params.vectors
        return cfg[DENSE].size if isinstance(cfg, dict) else None

    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        if recreate and self.exists():
            self.client.delete_collection(self.collection)
        if self.exists():
            existing = self.dense_size()
            if existing != dim:
                raise ValueError(
                    f"Collection {self.collection!r} has dense size {existing}, embeddings give "
                    f"{dim}. Re-run ingest with --full after changing the embedding model."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        with warnings.catch_warnings():
            # Embedded (local) Qdrant ignores payload indexes and warns; the server uses them.
            warnings.simplefilter("ignore", UserWarning)
            for field in ("doc_id", "lang"):
                self.client.create_payload_index(
                    self.collection, field, models.PayloadSchemaType.KEYWORD
                )

    # --- ingest side -----------------------------------------------------------------
    def indexed_documents(self) -> dict[str, str]:
        """doc_id -> ingest_key currently stored."""
        if not self.exists():
            return {}
        found: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id", "ingest_key"],
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                found[payload["doc_id"]] = payload.get("ingest_key", "")
            if offset is None:
                break
        return found

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: list[list[float]],
        sparse: list[models.SparseVector] | None,
        *,
        ingest_key: str,
        embedding_model: str,
    ) -> None:
        points = []
        for i, chunk in enumerate(chunks):
            vectors: dict[str, Any] = {DENSE: dense[i]}
            if sparse is not None:
                vectors[SPARSE] = sparse[i]
            points.append(
                models.PointStruct(
                    id=point_id(chunk.doc_id, chunk.chunk_index),
                    vector=vectors,
                    payload={
                        "text": chunk.text,
                        "doc_id": chunk.doc_id,
                        "source": chunk.source,
                        "section": chunk.section,
                        "lang": chunk.lang,
                        "updated_at": chunk.updated_at,
                        "content_hash": chunk.content_hash,
                        "chunk_index": chunk.chunk_index,
                        "ingest_key": ingest_key,
                        "embedding_model": embedding_model,
                    },
                )
            )
        if points:
            self.client.upsert(self.collection, points=points)

    def count(self) -> int:
        return self.client.count(self.collection).count if self.exists() else 0

    # --- query side ------------------------------------------------------------------
    def search(
        self,
        dense_query: list[float],
        sparse_query: models.SparseVector | None,
        *,
        limit: int,
        candidate_limit: int | None = None,
    ) -> list[Hit]:
        if not self.exists():
            return []
        candidates = candidate_limit or max(limit * 3, 12)
        if sparse_query is not None:
            response = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense_query, using=DENSE, limit=candidates),
                    models.Prefetch(query=sparse_query, using=SPARSE, limit=candidates),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=[DENSE],
            )
        else:
            response = self.client.query_points(
                self.collection,
                query=dense_query,
                using=DENSE,
                limit=limit,
                with_payload=True,
                with_vectors=[DENSE],
            )
        hits: list[Hit] = []
        for p in response.points:
            vec = p.vector.get(DENSE) if isinstance(p.vector, dict) else None
            hits.append(Hit(payload=p.payload or {}, dense_vector=vec, fusion_score=p.score))  # type: ignore[arg-type]
        return hits
