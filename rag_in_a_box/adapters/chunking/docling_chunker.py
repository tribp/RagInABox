from __future__ import annotations

import uuid
from typing import Any, Iterable, Optional

from rag_in_a_box.core.models import Chunk, Document


class DoclingChunker:
    """
    Chunker that uses Docling's HybridChunker + OpenAITokenizer (tiktoken),
    aligned to an OpenAI embeddings model (default: text-embedding-3-small).

    - If a DoclingDocument is already present in doc.metadata (recommended),
      we chunk it directly.
    - Otherwise we try to build a DoclingDocument from doc.uri (file/URL) or
      from doc.content via DocumentConverter.convert_string (MD/HTML).
    """

    def __init__(
        self,
        *,
        embed_model: str = "text-embedding-3-small",
        max_tokens: int = 8192,
        merge_peers: bool = True,
        use_contextualized_text: bool = True,
        # Where you stash a DoclingDocument inside Document.metadata:
        docling_document_keys: Iterable[str] = ("dl_doc", "docling_document", "docling_doc"),
    ):
        self.embed_model = embed_model
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.use_contextualized_text = use_contextualized_text
        self.docling_document_keys = tuple(docling_document_keys)

        # ---- Tokenizer (tiktoken) + Docling OpenAITokenizer wrapper ----
        import tiktoken
        from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer  # requires docling-core[chunking-openai]

        self._encoding = tiktoken.encoding_for_model(embed_model)
        self._tokenizer = OpenAITokenizer(tokenizer=self._encoding, max_tokens=max_tokens)

        # ---- HybridChunker ----
        try:
            # Most common when you install `docling`
            from docling.chunking import HybridChunker
        except Exception:
            # Fallback if you're using docling-core directly
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

        self._chunker = HybridChunker(tokenizer=self._tokenizer, merge_peers=merge_peers)

    def _count_tokens(self, text: str) -> int:
        # Ensure token counting matches the embedding model tokenizer
        return len(self._encoding.encode(text or ""))

    def _get_docling_document_from_metadata(self, doc: Document) -> Optional[Any]:
        md = doc.metadata if isinstance(doc.metadata, dict) else None
        if not md:
            return None
        for k in self.docling_document_keys:
            if k in md and md[k] is not None:
                return md[k]
        return None

    def _to_docling_document(self, doc: Document) -> Optional[Any]:
        """
        Best-effort conversion to a DoclingDocument:
        1) use an existing DoclingDocument from metadata
        2) else convert from doc.uri (file path / URL)
        3) else convert from doc.content as Markdown (or HTML if mime suggests)
        """
        dl_doc = self._get_docling_document_from_metadata(doc)
        if dl_doc is not None:
            return dl_doc

        # If you can re-convert from source (path/URL), this is the most faithful.
        if getattr(doc, "uri", None):
            try:
                from docling.document_converter import DocumentConverter
                return DocumentConverter().convert(source=doc.uri).document
            except Exception:
                pass

        # Otherwise, try from in-memory content (Docling supports MD/HTML strings).
        if getattr(doc, "content", None):
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.base_models import InputFormat

                fmt = InputFormat.HTML if (doc.mime_type or "").lower() in ("text/html", "application/xhtml+xml") else InputFormat.MD
                name = (doc.uri or doc.id or "document")
                return DocumentConverter().convert_string(content=doc.content, format=fmt, name=name).document
            except Exception:
                return None

        return None

    def chunk(self, doc: Document) -> list[Chunk]:
        dl_doc = self._to_docling_document(doc)

        # If we cannot obtain a DoclingDocument, return no chunks (or choose your own fallback).
        if dl_doc is None:
            return []

        chunks: list[Chunk] = []

        for i, dl_chunk in enumerate(self._chunker.chunk(dl_doc=dl_doc)):
            # Docling recommends embedding the context-enriched representation.
            # (Headings/captions are prepended where relevant.)
            if self.use_contextualized_text:
                text = self._chunker.contextualize(chunk=dl_chunk)
            else:
                text = getattr(dl_chunk, "text", "") or ""

            token_count = self._count_tokens(text)

            # Deterministic chunk id
            cid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}:{i}:{text}"))

            # Best-effort extraction of docling chunk metadata (field names differ across versions)
            dl_meta = getattr(dl_chunk, "meta", None)
            if dl_meta is None:
                dl_meta = getattr(dl_chunk, "metadata", None)

            metadata: dict[str, Any] = {
                **(doc.metadata or {}),
                "chunk_index": i,
                "mime_type": doc.mime_type,
                "tokens": token_count,
                "embed_model": self.embed_model,
                "docling_chunk_meta": dl_meta,
                # keep both forms if you want to debug retrieval quality:
                "docling_raw_text": getattr(dl_chunk, "text", None),
                "docling_contextualized": self.use_contextualized_text,
            }

            chunks.append(
                Chunk(
                    id=cid,
                    document_id=doc.id,
                    uri=doc.uri,
                    text=text,
                    metadata=metadata,
                )
            )

        return chunks
