from __future__ import annotations

import hashlib
import math


class DummyHashEmbedder:
    """
    Deterministic embedder for smoke testing WITHOUT external services.
    Produces a fixed-length vector (default=64) of a given text string in a deterministic way.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        if not text:
            return v

        # Hash into buckets
        for token in text.lower().split():
            h = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(h[:2], "big") % self.dim
            v[idx] += 1.0

        # L2 normalize (helps cosine similarity)
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]
