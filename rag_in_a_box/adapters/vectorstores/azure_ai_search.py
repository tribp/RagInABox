from __future__ import annotations

import hashlib
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from azure.search.documents.models import VectorizedQuery

from rag_in_a_box.core.models import Chunk, SearchResult


VECTOR_PROFILE = "raginabox-vector-profile"
HNSW_CONFIG = "raginabox-hnsw"


def _safe_domain(uri: str) -> str:
    try:
        return (urlparse(uri).netloc or "").lower()
    except Exception:
        return ""


@dataclass
class AzureAISearchVectorStore:
    endpoint: str
    api_key: str
    index_name: str
    vector_dim: int

    def __post_init__(self) -> None:
        cred = AzureKeyCredential(self.api_key)
        self._index_client = SearchIndexClient(self.endpoint, cred)
        self._search_client = SearchClient(self.endpoint, self.index_name, cred)

    def ensure_index(self) -> None:
        fields = [
            # Core
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="uri", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="referrer_url", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="chunk_start_char", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="referrer_url", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="mime_type", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="content_hash", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="ingested_at", type=SearchFieldDataType.DateTimeOffset, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),

            # Vector field
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.vector_dim,
                vector_search_profile_name=VECTOR_PROFILE,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name=HNSW_CONFIG)],
            profiles=[VectorSearchProfile(name=VECTOR_PROFILE, algorithm_configuration_name=HNSW_CONFIG)],
        )

        index = SearchIndex(name=self.index_name, fields=fields, vector_search=vector_search)
        self._index_client.create_or_update_index(index)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        docs = []
        for i, (c, v) in enumerate(zip(chunks, vectors)):
            metadata = c.metadata or {}
            start_char = metadata.get("chunk_start_char", metadata.get("start_char"))
            source_type = metadata.get("source_type", metadata.get("source"))
            domain = urlparse(c.uri).netloc.lower() if c.uri else None
            content_hash = hashlib.sha256(c.text.encode("utf-8")).hexdigest()

            docs.append(
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "uri": c.uri,
                    "chunk_index": metadata.get("chunk_index", i),
                    "chunk_start_char": start_char,
                    "referrer_url": metadata.get("referrer_url"),
                    "source_type": source_type,
                    "mime_type": metadata.get("mime_type"),
                    "domain": domain,
                    "content_hash": content_hash,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "title": metadata.get("title"),
                    "content": c.text,
                    "content_vector": v,
                }
            )

        self._search_client.upload_documents(documents=docs)

    def query(self, query_vector: list[float], k: int, filters: Optional[dict] = None) -> list[SearchResult]:
        vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="content_vector", kind="vector")

        results = self._search_client.search(
            vector_queries=[vq],
            select=[
                "id",
                "document_id",
                "uri",
                "referrer_url",
                "chunk_index",
                "title",
                "source_type",
                "mime_type",
                "domain",
                "ingested_at",
                "content_hash",
                "chunk_start_char",
                "content",
            ],
            top=k,
        )

        out: list[SearchResult] = []
        for r in results:
            chunk = Chunk(
                id=r["id"],
                document_id=r["document_id"],
                uri=r["uri"],
                text=r["content"],
                metadata={
                    "chunk_index": r.get("chunk_index"),
                     "referrer_url": r.get("referrer_url"),
                    "title": r.get("title"),
                    "source_type": r.get("source_type"),
                    "mime_type": r.get("mime_type"),
                    "domain": r.get("domain"),
                    "ingested_at": r.get("ingested_at"),
                    "content_hash": r.get("content_hash"),
                    "start_char": r.get("chunk_start_char"),
                },
            )
            out.append(SearchResult(chunk=chunk, score=r.get("@search.score", 0.0)))

        return out
