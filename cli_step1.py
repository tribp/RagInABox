from __future__ import annotations

from rag_in_a_box.config.settings import Settings


def main() -> None:
    s = Settings()
    print("✅ Settings loaded")
    print(f"Azure OpenAI endpoint: {s.azure_openai_endpoint}")
    print(f"Chat deployment:       {s.azure_openai_chat_deployment}")
    print(f"Embedding deployment:  {s.azure_openai_embedding_deployment}")
    print(f"Search endpoint:       {s.azure_search_endpoint}")
    print(f"Search index:          {s.azure_search_index}")
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
    print(f"Top K:                 {s.top_k}")


if __name__ == "__main__":
    main()
