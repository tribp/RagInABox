from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        # Vector index creation pattern matches Azure vector search quickstart :contentReference[oaicite:6]{index=6}
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="uri", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
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
            docs.append(
                {
                    "id": c.id,
                    "document_id": c.document_id,
                    "uri": c.uri,
                    "chunk_index": c.metadata.get("chunk_index", i),
                    "content": c.text,
                    "content_vector": v,
                }
            )
        self._search_client.upload_documents(documents=docs)

    def query(self, query_vector: list[float], k: int, filters: Optional[dict] = None) -> list[SearchResult]:
        # Vector query pattern matches quickstart: VectorizedQuery + vector_queries=[...] :contentReference[oaicite:7]{index=7}
        vq = VectorizedQuery(vector=query_vector, k_nearest_neighbors=k, fields="content_vector", kind="vector")
        results = self._search_client.search(
            vector_queries=[vq],
            select=["id", "document_id", "uri", "chunk_index", "content"],
            top=k,
        )

        out: list[SearchResult] = []
        for r in results:
            chunk = Chunk(
                id=r["id"],
                document_id=r["document_id"],
                uri=r["uri"],
                text=r["content"],
                metadata={"chunk_index": r.get("chunk_index")},
            )
            out.append(SearchResult(chunk=chunk, score=r.get("@search.score", 0.0)))
        return out
