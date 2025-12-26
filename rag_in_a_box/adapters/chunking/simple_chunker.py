from __future__ import annotations

import uuid

from rag_in_a_box.core.models import Chunk, Document


class SimpleCharChunker:
    """
    Simple character-based chunking with overlap.
    Works for any extracted text. Later we can add markdown-aware chunkers.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        text = (doc.content or "").strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        i = 0
        chunk_index = 0

        while i < len(text):
            window = text[i : i + self.chunk_size]
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}:{chunk_index}:{i}"))
            chunks.append(
                Chunk(
                    id=cid,
                    document_id=doc.id,
                    uri=doc.uri,
                    text=window,
                    metadata={
                        **doc.metadata,
                        "chunk_index": chunk_index,
                        "start_char": i,
                    },
                )
            )
            chunk_index += 1
            i += step

        return chunks
