from __future__ import annotations

import hashlib
from dataclasses import dataclass

from bs4 import BeautifulSoup

from rag_in_a_box.core.models import Document, SourceItem


@dataclass
class HtmlExtractor:
    def can_handle(self, mime_type: str) -> bool:
        return mime_type in {"text/html", "application/xhtml+xml"}

    def extract(self, item: SourceItem) -> Document:
        html = item.data.decode("utf-8", errors="ignore") if isinstance(item.data, bytes) else item.data
        soup = BeautifulSoup(html, "html.parser")

        # Remove scripts/styles for cleaner text
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        page_title = (soup.title.string or "") if soup.title else ""
        metadata = {**item.metadata}
        if page_title := page_title.strip():
            metadata["title"] = page_title

        doc_id = hashlib.sha256(item.uri.encode("utf-8")).hexdigest()
        return Document(
            id=doc_id,
            uri=item.uri,
            content=text,
            mime_type=item.mime_type,
            metadata=metadata,
        )
