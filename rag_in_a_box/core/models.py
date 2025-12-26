from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


Metadata = Dict[str, Any]


@dataclass(frozen=True)
class SourceItem:
    """
    Raw item produced by a ContentSource.
    - uri: where it came from (file path, URL, blob URL, etc.)
    - data: either bytes (e.g., pdf) or text
    - mime_type: used to select an extractor
    """
    uri: str
    data: bytes | str
    mime_type: str = "text/plain"
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    """
    Extracted, readable text with metadata.
    """
    id: str
    uri: str
    content: str
    mime_type: str = "text/plain"
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """
    A chunk of text derived from a Document.
    """
    id: str
    document_id: str
    uri: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class ChatResponse:
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
