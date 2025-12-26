from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from rag_in_a_box.core.models import Chunk, SearchResult


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a)) or 1.0


def _cosine(a: list[float], b: list[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


@dataclass
class _Row:
    chunk: Chunk
    vector: list[float]


class InMemoryVectorStore:
    """
    Minimal VectorStore for local testing.
    """

    def __init__(self):
        self._rows: list[_Row] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        # simple append; later we’ll do true upsert by id
        for c, v in zip(chunks, vectors):
            self._rows.append(_Row(chunk=c, vector=v))

    def query(
        self,
        query_vector: list[float],
        k: int,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        # filters are ignored in this toy implementation
        scored = [
            SearchResult(chunk=row.chunk, score=_cosine(query_vector, row.vector))
            for row in self._rows
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
