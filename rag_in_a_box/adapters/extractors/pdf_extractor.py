from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader

from rag_in_a_box.core.models import Document, SourceItem


@dataclass
class PdfExtractor:
    def can_handle(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    def extract(self, item: SourceItem) -> Document:
        if not isinstance(item.data, bytes):
            raise TypeError("PdfExtractor expects bytes data")

        reader = PdfReader(BytesIO(item.data))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        text = "\n".join(pages).strip()

        doc_id = hashlib.sha256(item.uri.encode("utf-8")).hexdigest()
        return Document(
            id=doc_id,
            uri=item.uri,
            content=text,
            mime_type=item.mime_type,
            metadata=item.metadata,
        )
