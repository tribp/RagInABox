from __future__ import annotations

import argparse

from rag_in_a_box.config.settings import Settings
from rag_in_a_box.adapters.chunking.simple_chunker import SimpleCharChunker
from rag_in_a_box.adapters.chunking.docling_chunker import DoclingChunker
from rag_in_a_box.adapters.extractors.registry import ExtractorRegistry
from rag_in_a_box.adapters.extractors.text_extractor import TextExtractor
from rag_in_a_box.adapters.extractors.html_extractor import HtmlExtractor
from rag_in_a_box.adapters.extractors.pdf_extractor import PdfExtractor
from rag_in_a_box.adapters.extractors.docling_extractor import DoclingExtractor
from rag_in_a_box.adapters.sources.local_folder import LocalFolderSource
from rag_in_a_box.adapters.embeddings.azure_openai import AzureOpenAIEmbedder
from rag_in_a_box.adapters.vectorstores.azure_ai_search import AzureAISearchVectorStore
from rag_in_a_box.pipelines.ingestion import IngestionPipeline
from rag_in_a_box.adapters.sources.website_crawler import WebsiteCrawlerSource



def _build_chunker(args, settings: Settings):
    if getattr(args, "use_docling_chunker", False):
        return DoclingChunker()
    return SimpleCharChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)


def cmd_ingest_local(args) -> None:
    s = Settings()

    embedder = AzureOpenAIEmbedder(
        endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        deployment=s.azure_openai_embedding_deployment,
    )

    vector_dim = embedder.embedding_dim()
    store = AzureAISearchVectorStore(
        endpoint=s.azure_search_endpoint,
        api_key=s.azure_search_api_key,
        index_name=s.azure_search_index,
        vector_dim=vector_dim,
    )
    store.ensure_index()

    source = LocalFolderSource(root=s.ingest_local_path, recursive=True)

    registry = ExtractorRegistry(
        extractors=[
            DoclingExtractor(),
            PdfExtractor(),
            HtmlExtractor(),
            TextExtractor(),
        ]
    )

    chunker = _build_chunker(args, s)

    pipeline = IngestionPipeline(
        source=source,
        extractor_registry=registry,
        chunker=chunker,
        embedder=embedder,
        vectorstore=store,
        batch_size=64,
    )

    stats = pipeline.run()
    print("✅ Ingestion complete")
    print(stats)

def cmd_ingest_web(args) -> None:
    s = Settings()

    embedder = AzureOpenAIEmbedder(
        endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        deployment=s.azure_openai_embedding_deployment,
    )

    vector_dim = embedder.embedding_dim()
    store = AzureAISearchVectorStore(
        endpoint=s.azure_search_endpoint,
        api_key=s.azure_search_api_key,
        index_name=s.azure_search_index,
        vector_dim=vector_dim,
    )
    store.ensure_index()

    source = WebsiteCrawlerSource(
        start_urls=s.ingest_start_urls,
        allowed_domains=s.ingest_allowed_domains,
        exclude_prefixes=s.crawl_exclude_prefixes,
        max_pages=s.crawl_max_pages,
        max_depth=s.crawl_max_depth,
        timeout_seconds=s.crawl_timeout_seconds,
        user_agent=s.crawl_user_agent,
        include_pdfs=s.crawl_include_pdfs,
    )

    registry = ExtractorRegistry(
        extractors=[
            DoclingExtractor(),
            PdfExtractor(),
            HtmlExtractor(),
            TextExtractor(),
        ]
    )

    chunker = _build_chunker(args, s)

    pipeline = IngestionPipeline(
        source=source,
        extractor_registry=registry,
        chunker=chunker,
        embedder=embedder,
        vectorstore=store,
        batch_size=16,      #64 resulted in error
    )

    stats = pipeline.run()
    print("✅ Web ingestion complete")
    print(stats)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-in-a-box")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_local_parser = sub.add_parser(
        "ingest-local", help="Ingest files from INGEST_LOCAL_PATH into Azure AI Search"
    )
    ingest_web_parser = sub.add_parser(
        "ingest-web", help="Crawl INGEST_START_URLS and index pages + PDFs into Azure AI Search"
    )

    for p in (ingest_local_parser, ingest_web_parser):
        p.add_argument(
            "--docling-chunker",
            dest="use_docling_chunker",
            action="store_true",
            help="Use DoclingChunker instead of the default character chunker",
        )

    args = parser.parse_args()
    
    if args.cmd == "ingest-local":
        cmd_ingest_local(args)
    elif args.cmd == "ingest-web":
        cmd_ingest_web(args)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
