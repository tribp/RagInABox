# RagInABox

RagInABox is a minimal but composable retrieval-augmented generation starter. It focuses on:

- Structured extraction with Docling so you preserve layout metadata for smarter chunking.
- Interchangeable chunkers (Docling-aware or simple character windows) that guard against oversized inputs.
- Azure OpenAI embeddings feeding Azure AI Search vector indexes.
- Local folder and website crawling sources wired into a shared ingestion pipeline.
- A small RAG chat engine that reuses the same interfaces.

Former step-by-step scaffolding has been removed in favor of a concise, production-minded layout.

## Architecture

- **Core models & interfaces** (`rag_in_a_box/core`): typed dataclasses and Protocols for sources, extractors, chunkers, embedders, vector stores, and LLMs.
- **Adapters** (`rag_in_a_box/adapters`):
  - Sources: local folder walker and HTTP crawler.
  - Extractors: Docling (structured), PDF/HTML/text fallbacks.
  - Chunkers: Docling-aware chunker and simple character-based chunker.
  - Embeddings: Azure OpenAI.
  - Vector stores: Azure AI Search (and in-memory for tests).
  - LLM: Azure OpenAI chat helper.
- **Pipelines** (`rag_in_a_box/pipelines`): ingestion (source → extract → chunk → embed → store) and chat retrieval/generation.
- **CLI** (`rag_in_a_box/cli.py`): commands to ingest local files or crawl websites with shared wiring.

## Configuration

Configuration is centralized in `rag_in_a_box/config/settings.py` (Pydantic settings). Provide values through environment variables:

| Variable | Purpose |
| --- | --- |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | API key for the OpenAI resource |
| `AZURE_OPENAI_API_VERSION` | API version (e.g., `2024-xx-xx`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name for chat completions |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name for embeddings |
| `EMBEDDING_REQUESTS_PER_MINUTE` | Optional throttle for embedding calls |
| `EMBEDDING_MAX_CONCURRENCY` | Max concurrent embedding requests |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_API_KEY` | Admin/query key for Azure AI Search |
| `AZURE_SEARCH_INDEX` | Vector index name |
| `INGEST_LOCAL_PATH` | Default folder to ingest when using `ingest-local` |
| `INGEST_START_URLS` | Comma-separated list of starting URLs for crawling |
| `INGEST_ALLOWED_DOMAINS` | Allowed domains for the crawler (comma-separated) |
| `CRAWL_MAX_PAGES` | Crawl page budget |
| `CRAWL_MAX_DEPTH` | Depth limit for link following |
| `CRAWL_TIMEOUT_SECONDS` | HTTP timeout for fetches |
| `CRAWL_USER_AGENT` | User agent string used by the crawler |
| `CRAWL_INCLUDE_PDFS` | Whether to download PDFs during crawling |
| `CRAWL_EXCLUDE_PREFIXES` | Comma-separated URL prefixes to skip |
| `CHUNK_SIZE` | Size for the simple character chunker |
| `CHUNK_OVERLAP` | Overlap for the simple character chunker |
| `TOP_K` | Retrieval depth for chat |

## Setup

1. Install dependencies (via [uv](https://docs.astral.sh/uv/) or pip):

   ```bash
   uv sync
   # or
   pip install -e .
   ```

2. Export the environment variables above (or create a `.env` file recognized by Pydantic Settings).

## Ingestion CLI

Two commands share the same ingestion pipeline wiring and use Docling by default:

```bash
uv run python -m rag_in_a_box.cli ingest-local  # index files under INGEST_LOCAL_PATH
uv run python -m rag_in_a_box.cli ingest-web    # crawl INGEST_START_URLS and index pages/PDFs
```

Switch to the character-based chunker when Docling metadata is unavailable or undesired:

```bash
uv run python -m rag_in_a_box.cli ingest-local --chunker simplechunker
```

## Running tests

```bash
uv run pytest
```

Tests include Docling-focused adapters; when Docling is not installed they rely on the injected fake converter used in the test suite.
