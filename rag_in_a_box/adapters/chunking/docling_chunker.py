from __future__ import annotations

import uuid
from typing import Any

from rag_in_a_box.core.models import Chunk, Document


class DoclingChunker:
    """Chunker that uses Docling structural cues instead of raw character windows."""

    def chunk(self, doc: Document) -> list[Chunk]:
        docling = doc.metadata.get("docling") if isinstance(doc.metadata, dict) else None
        sections: list[dict[str, Any]] = []
        if docling and isinstance(docling, dict):
            sections = docling.get("sections") or []

        if not sections:
            # Fallback to a single chunk when structural information is missing
            return [
                Chunk(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}:doc")),
                    document_id=doc.id,
                    uri=doc.uri,
                    text=doc.content,
                    metadata={**doc.metadata, "chunk_index": 0},
                )
            ] if doc.content else []

        chunks: list[Chunk] = []
        chunk_index = 0
        for section in sections:
            section_id = section.get("id")
            section_title = section.get("title")
            for para in section.get("paragraphs", []):
                para_text = str(para.get("text", "")).strip()
                if not para_text:
                    continue
                para_id = para.get("id")
                page_number = para.get("page")
                cid = str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"{doc.id}:{section_id}:{para_id}:{chunk_index}",
                    )
                )
                metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_index,
                    "section_id": section_id,
                    "section_title": section_title,
                    "paragraph_id": para_id,
                }
                if page_number is not None:
                    metadata["page_number"] = page_number

                chunks.append(
                    Chunk(
                        id=cid,
                        document_id=doc.id,
                        uri=doc.uri,
                        text=para_text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

        return chunks
