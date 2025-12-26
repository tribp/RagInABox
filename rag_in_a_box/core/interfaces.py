from __future__ import annotations

from typing import Iterable, Optional, Protocol, runtime_checkable

from rag_in_a_box.core.models import Chunk, ChatMessage, Document, SearchResult, SourceItem


@runtime_checkable
class ContentSource(Protocol):
    def iter_items(self) -> Iterable[SourceItem]:
        ...


@runtime_checkable
class Extractor(Protocol):
    def can_handle(self, mime_type: str) -> bool:
        ...

    def extract(self, item: SourceItem) -> Document:
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, doc: Document) -> list[Chunk]:
        ...


@runtime_checkable
class Embedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        ...

    def query(
        self,
        query_vector: list[float],
        k: int,
        filters: Optional[dict] = None,
    ) -> list[SearchResult]:
        ...


@runtime_checkable
class LLM(Protocol):
    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        ...
