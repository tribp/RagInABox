from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from rag_in_a_box.core.models import Document, SourceItem


@dataclass
class DoclingExtractor:
    """Extractor that delegates parsing to Docling.

    The extractor stores Docling's structured metadata on ``Document.metadata``
    so downstream chunkers can leverage layout hints.
    """

    converter: Any | None = None
    _supported_types: set[str] = field(
        default_factory=lambda: {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "application/vnd.ms-powerpoint",
        }
    )

    def can_handle(self, mime_type: str) -> bool:
        return mime_type in self._supported_types

    def _get_converter(self):
        if self.converter is not None:
            return self.converter

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                "Docling must be installed to use DoclingExtractor"
            ) from exc

        self.converter = DocumentConverter()
        return self.converter

    def extract(self, item: SourceItem) -> Document:
        if not isinstance(item.data, bytes):
            raise TypeError("DoclingExtractor expects bytes data")

        converter = self._get_converter()
        if hasattr(converter, "convert_bytes"):
            conversion = converter.convert_bytes(item.data, mime_type=item.mime_type)
        elif hasattr(converter, "convert"):
            conversion = converter.convert(item.data)
        else:  # pragma: no cover - defensive
            raise AttributeError("Converter does not support conversion methods")

        docling_payload = self._serialize_docling_result(conversion)
        text_content = self._render_text(conversion, docling_payload)

        doc_id = hashlib.sha256(item.uri.encode("utf-8")).hexdigest()
        metadata = {**item.metadata, "docling": docling_payload}

        return Document(
            id=doc_id,
            uri=item.uri,
            content=text_content,
            mime_type=item.mime_type,
            metadata=metadata,
        )

    def _serialize_docling_result(self, conversion: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metadata": getattr(conversion, "metadata", {}),
            "sections": [],
        }

        sections = getattr(conversion, "sections", None)
        if sections:
            payload["sections"] = [
                self._serialize_section(sec, idx) for idx, sec in enumerate(sections)
            ]
            return payload

        document = getattr(conversion, "document", None)
        if document and hasattr(document, "sections"):
            payload["sections"] = [
                self._serialize_section(sec, idx)
                for idx, sec in enumerate(getattr(document, "sections"))
            ]

        return payload

    def _serialize_section(self, section: Any, index: int) -> dict[str, Any]:
        title = getattr(section, "title", None) or getattr(section, "heading", None)
        paragraphs = getattr(section, "paragraphs", None) or []
        serialized_paragraphs: list[dict[str, Any]] = []
        for para_idx, para in enumerate(paragraphs):
            serialized_paragraphs.append(
                {
                    "id": getattr(para, "id", f"p{para_idx}"),
                    "text": getattr(para, "text", ""),
                    "page": getattr(para, "page", None),
                }
            )

        return {
            "id": getattr(section, "id", f"s{index}"),
            "title": title,
            "paragraphs": serialized_paragraphs,
        }

    def _render_text(self, conversion: Any, docling_payload: dict[str, Any]) -> str:
        section_data = docling_payload.get("sections") or []
        if section_data:
            lines: list[str] = []
            for sec in section_data:
                if sec.get("title"):
                    lines.append(str(sec["title"]).strip())
                for para in sec.get("paragraphs", []):
                    para_text = str(para.get("text", "")).strip()
                    if para_text:
                        lines.append(para_text)
            return "\n\n".join(lines).strip()

        if hasattr(conversion, "text"):
            text = getattr(conversion, "text")
            if callable(text):
                text = text()
            if isinstance(text, str) and text.strip():
                return text.strip()

        document = getattr(conversion, "document", None)
        if document:
            if hasattr(document, "export_to_text"):
                exported = document.export_to_text()
                if isinstance(exported, str) and exported.strip():
                    return exported.strip()
            if hasattr(document, "text"):
                doc_text = document.text
                if isinstance(doc_text, str) and doc_text.strip():
                    return doc_text.strip()

        return ""
