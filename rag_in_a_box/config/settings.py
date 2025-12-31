from __future__ import annotations

from typing import Iterable

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_csv_list(raw: Iterable[str] | str | None) -> list[str]:
    """Accept comma-separated strings (from .env) or list-like values."""

    if raw is None or raw == "":
        return []

    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)

    return [str(val).strip() for val in values if str(val).strip()]


class Settings(BaseSettings):
    """
    Typed settings loaded from .env (and environment variables).
    Keep this as the single source of truth for configuration keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # -------------------
    # Azure OpenAI
    # -------------------
    azure_openai_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str = Field(..., alias="AZURE_OPENAI_API_VERSION")
    azure_openai_chat_deployment: str = Field(..., alias="AZURE_OPENAI_CHAT_DEPLOYMENT")
    azure_openai_embedding_deployment: str = Field(..., alias="AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    embedding_requests_per_minute: int | None = Field(
        None, alias="EMBEDDING_REQUESTS_PER_MINUTE"
    )
    embedding_max_concurrency: int = Field(5, alias="EMBEDDING_MAX_CONCURRENCY")

    # -------------------
    # Azure AI Search
    # -------------------
    azure_search_endpoint: str = Field(..., alias="AZURE_SEARCH_ENDPOINT")
    azure_search_api_key: str = Field(..., alias="AZURE_SEARCH_API_KEY")
    azure_search_index: str = Field("raginabox", alias="AZURE_SEARCH_INDEX")
    azure_search_batch_size: int | None = Field(
        16, alias="AZURE_SEARCH_BATCH_SIZE")

    # -------------------
    # Ingestion defaults
    # -------------------
    # Local folder ingestion
    ingest_local_path: str = Field("./data", alias="INGEST_LOCAL_PATH")
    
    # Multiple start URLs + allowed domains (comma-separated in .env)
    ingest_start_urls: list[str] = Field(default_factory=list, alias="INGEST_START_URLS")
    ingest_allowed_domains: list[str] = Field(default_factory=list, alias="INGEST_ALLOWED_DOMAINS")

    @field_validator("ingest_start_urls", mode="before")
    @classmethod
    def _parse_start_urls(cls, v):
        return _parse_csv_list(v)

    @field_validator("ingest_allowed_domains", mode="before")
    @classmethod
    def _parse_allowed_domains(cls, v):
        return [d.lower() for d in _parse_csv_list(v)]
    
    # -------------------
    # Web crawling
    # -------------------
    crawl_max_pages: int = Field(200, alias="CRAWL_MAX_PAGES")
    crawl_max_depth: int = Field(3, alias="CRAWL_MAX_DEPTH")
    crawl_timeout_seconds: int = Field(20, alias="CRAWL_TIMEOUT_SECONDS")
    crawl_user_agent: str = Field("RagInABoxBot/0.1", alias="CRAWL_USER_AGENT")
    crawl_include_pdfs: bool = Field(True, alias="CRAWL_INCLUDE_PDFS")
    # Exclude URLs by prefix (comma-separated in .env), e.g. https://www.fluvius.be/fr
    crawl_exclude_prefixes: list[str] = Field(default_factory=list, alias="CRAWL_EXCLUDE_PREFIXES")

    @field_validator("crawl_exclude_prefixes", mode="before")
    @classmethod
    def _parse_exclude_prefixes(cls, value):
        # Normalize by stripping whitespace and trailing slash
        return [prefix.rstrip("/") for prefix in _parse_csv_list(value)]

    @field_validator("embedding_max_concurrency")
    @classmethod
    def _positive_concurrency(cls, v):
        if v <= 0:
            raise ValueError("EMBEDDING_MAX_CONCURRENCY must be greater than 0")
        return v

    @field_validator("embedding_requests_per_minute")
    @classmethod
    def _non_negative_rpm(cls, v):
        if v is None:
            return None
        if v <= 0:
            raise ValueError("EMBEDDING_REQUESTS_PER_MINUTE must be positive when set")
        return v

    # -------------------
    # Chunking + retrieval
    # -------------------
    chunk_size: int = Field(800, alias="CHUNK_SIZE")
    chunk_max_tokens: int = Field(800, alias="CHUNK_MAX_TOKENS")
    chunk_overlap: int = Field(120, alias="CHUNK_OVERLAP")
    top_k: int = Field(5, alias="TOP_K")
    
if __name__ == "__main__":
    s = Settings()
    print("✅ Settings loaded")
    print(f"Azure OpenAI endpoint: {s.azure_openai_endpoint}")
    print(f"Chat deployment:       {s.azure_openai_chat_deployment}")
    print(f"Embedding deployment:  {s.azure_openai_embedding_deployment}")
    print(f"Search endpoint:       {s.azure_search_endpoint}")
    print(f"Search index:          {s.azure_search_index}")
    print(f"Search batch size:     {s.azure_search_batch_size}")
    print(f"Ingest local path:     {s.ingest_local_path}")
    print(f"Start URLs:            {s.ingest_start_urls}")
    print(f"Allowed domains:       {s.ingest_allowed_domains}")
    print(f"Crawl exclude prefixes: {s.crawl_exclude_prefixes}")
    print(f"Crawl max pages:       {s.crawl_max_pages}")
    print(f"Crawl max depth:       {s.crawl_max_depth}")
    print(f"Crawl timeout (s):     {s.crawl_timeout_seconds}")
    print(f"Crawl user agent:      {s.crawl_user_agent}")
    print(f"Crawl include PDFs:    {s.crawl_include_pdfs}") 
    print(f"Chunk size/overlap:    {s.chunk_size}/{s.chunk_overlap}")
    print(f"Chunk max tokens:      {s.chunk_max_tokens}")
    print(f"Top K:                 {s.top_k}")
