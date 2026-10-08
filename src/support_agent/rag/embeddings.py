"""Local multilingual embeddings via fastembed (ONNX, CPU), with normalisation.

fastembed returns *un-normalised* vectors for e5 models, so cosine thresholds would be
meaningless without L2 normalisation here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FastEmbedEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        cache_dir: Path | str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self._model: Any = None
        # e5 models were trained with these prefixes; omitting them degrades retrieval.
        self._e5 = "e5" in model_name.lower()

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name, cache_dir=self.cache_dir)
        return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        return [_normalise(v.tolist()) for v in model.embed(texts, batch_size=self.batch_size)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefix = "passage: " if self._e5 else ""
        return self._embed([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        prefix = "query: " if self._e5 else ""
        return self._embed([prefix + text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
