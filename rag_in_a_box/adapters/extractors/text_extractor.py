from __future__ import annotations

import hashlib
from dataclasses import dataclass

from rag_in_a_box.core.models import Document, SourceItem


@dataclass
class TextExtractor:
    def can_handle(self, mime_type: str) -> bool:
        return mime_type.startswith("text/") or mime_type in {"application/json", "application/xml"}

    def extract(self, item: SourceItem) -> Document:
        if isinstance(item.data, bytes):
            content = item.data.decode("utf-8", errors="ignore")
        else:
            content = item.data

        doc_id = hashlib.sha256(item.uri.encode("utf-8")).hexdigest()
        return Document(
            id=doc_id,
            uri=item.uri,
            content=content,
            mime_type=item.mime_type,
            metadata=item.metadata,
        )
