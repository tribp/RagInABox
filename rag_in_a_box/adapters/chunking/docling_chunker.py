from __future__ import annotations

import uuid
from typing import Any

from rag_in_a_box.core.models import Chunk, Document


class DoclingChunker:
    """Chunker that uses Docling structural cues and guards against oversized inputs."""

    def __init__(self, max_chunk_chars: int = 8000, chunk_overlap: int = 200):
        if chunk_overlap >= max_chunk_chars:
            raise ValueError("chunk_overlap must be smaller than max_chunk_chars")
        self.max_chunk_chars = max_chunk_chars
        self.chunk_overlap = chunk_overlap

    def _append_chunks(
        self,
        *,
        doc: Document,
        text: str,
        base_metadata: dict[str, Any],
        chunks: list[Chunk],
        chunk_index: int,
    ) -> int:
        """Split text into windowed chunks to stay within model limits."""

        step = self.max_chunk_chars - self.chunk_overlap
        start = 0

        while start < len(text):
            window = text[start : start + self.max_chunk_chars]
            cid = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{doc.id}:{base_metadata.get('section_id')}:"
                    f"{base_metadata.get('paragraph_id')}:{chunk_index}:{start}",
                )
            )
            metadata = {**doc.metadata, **base_metadata, "chunk_index": chunk_index}
            if start:
                metadata["start_char"] = start

            chunks.append(
                Chunk(
                    id=cid,
                    document_id=doc.id,
                    uri=doc.uri,
                    text=window,
                    metadata=metadata,
                )
            )

            start += step
            chunk_index += 1

        return chunk_index

    def chunk(self, doc: Document) -> list[Chunk]:
        docling = doc.metadata.get("docling") if isinstance(doc.metadata, dict) else None
        sections: list[dict[str, Any]] = []
        if docling and isinstance(docling, dict):
            sections = docling.get("sections") or []

        chunks: list[Chunk] = []
        chunk_index = 0

        if not sections:
            if doc.content:
                chunk_index = self._append_chunks(
                    doc=doc,
                    text=doc.content,
                    base_metadata={},
                    chunks=chunks,
                    chunk_index=chunk_index,
                )
            return chunks

        for section in sections:
            section_id = section.get("id")
            section_title = section.get("title")
            for para in section.get("paragraphs", []):
                para_text = str(para.get("text", "")).strip()
                if not para_text:
                    continue

                para_id = para.get("id")
                page_number = para.get("page")
                metadata = {
                    "section_id": section_id,
                    "section_title": section_title,
                    "paragraph_id": para_id,
                }
                if page_number is not None:
                    metadata["page_number"] = page_number

                chunk_index = self._append_chunks(
                    doc=doc,
                    text=para_text,
                    base_metadata=metadata,
                    chunks=chunks,
                    chunk_index=chunk_index,
                )

        return chunks
