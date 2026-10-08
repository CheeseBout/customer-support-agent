"""Sparse (BM25) encoder for the keyword half of hybrid search."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import models


class SparseEncoder:
    def __init__(
        self, model_name: str = "Qdrant/bm25", cache_dir: Path | str | None = None
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self.model_name, cache_dir=self.cache_dir)
        return self._model

    @staticmethod
    def _to_vector(emb: Any) -> models.SparseVector:
        return models.SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())

    def encode_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [self._to_vector(e) for e in self._load().embed(texts)]

    def encode_query(self, text: str) -> models.SparseVector:
        return self._to_vector(next(iter(self._load().query_embed(text))))
