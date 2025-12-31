from __future__ import annotations

import argparse

from rag_in_a_box.adapters.chunking.docling_chunker import DoclingChunker
from rag_in_a_box.adapters.chunking.simple_chunker import SimpleCharChunker
from rag_in_a_box.adapters.embeddings.azure_openai import AzureOpenAIEmbedder
from rag_in_a_box.adapters.extractors.docling_extractor import DoclingExtractor
from rag_in_a_box.adapters.extractors.html_extractor import HtmlExtractor
from rag_in_a_box.adapters.extractors.pdf_extractor import PdfExtractor
from rag_in_a_box.adapters.extractors.registry import ExtractorRegistry
from rag_in_a_box.adapters.extractors.text_extractor import TextExtractor
from rag_in_a_box.adapters.sources.local_folder import LocalFolderSource
from rag_in_a_box.adapters.sources.website_crawler import WebsiteCrawlerSource
from rag_in_a_box.adapters.vectorstores.azure_ai_search import AzureAISearchVectorStore
from rag_in_a_box.config.settings import Settings
from rag_in_a_box.pipelines.ingestion import IngestionPipeline


def _build_chunker(args, settings: Settings):
    choice = (getattr(args, "chunker", None) or "doclingchunker").lower()
    if choice == "doclingchunker":
        return DoclingChunker()
    if choice == "simplechunker":
        return SimpleCharChunker(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
    raise SystemExit(f"Unknown chunker: {choice}")


def _build_embedder(settings: Settings) -> AzureOpenAIEmbedder:
    return AzureOpenAIEmbedder(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        deployment=settings.azure_openai_embedding_deployment,
        requests_per_minute=settings.embedding_requests_per_minute,
        max_concurrency=settings.embedding_max_concurrency,
    )


def _build_vectorstore(settings: Settings, embedder: AzureOpenAIEmbedder) -> AzureAISearchVectorStore:
    store = AzureAISearchVectorStore(
        endpoint=settings.azure_search_endpoint,
        api_key=settings.azure_search_api_key,
        index_name=settings.azure_search_index,
        vector_dim=embedder.embedding_dim(),
    )
    store.ensure_index()
    return store


def _build_registry() -> ExtractorRegistry:
    return ExtractorRegistry(
        extractors=[DoclingExtractor(), PdfExtractor(), HtmlExtractor(), TextExtractor()]
    )


def _run_pipeline(*, source, settings: Settings, chunker) -> None:
    embedder = _build_embedder(settings)
    vectorstore = _build_vectorstore(settings, embedder)
    pipeline = IngestionPipeline(
        source=source,
        extractor_registry=_build_registry(),
        chunker=chunker,
        embedder=embedder,
        vectorstore=vectorstore,
        batch_size=settings.azure_search_batch_size,
    )
    stats = pipeline.run()
    print("✅ Ingestion complete")
    print(stats)


def cmd_ingest_local(args) -> None:
    settings = Settings()
    source = LocalFolderSource(root=settings.ingest_local_path, recursive=True)
    chunker = _build_chunker(args, settings)
    _run_pipeline(source=source, settings=settings, chunker=chunker)


def cmd_ingest_web(args) -> None:
    settings = Settings()
    source = WebsiteCrawlerSource(
        start_urls=settings.ingest_start_urls,
        allowed_domains=settings.ingest_allowed_domains,
        exclude_prefixes=settings.crawl_exclude_prefixes,
        max_pages=settings.crawl_max_pages,
        max_depth=settings.crawl_max_depth,
        timeout_seconds=settings.crawl_timeout_seconds,
        user_agent=settings.crawl_user_agent,
        include_pdfs=settings.crawl_include_pdfs,
    )
    chunker = _build_chunker(args, settings)
    _run_pipeline(source=source, settings=settings, chunker=chunker)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-in-a-box")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest_local_parser = sub.add_parser(
        "ingest-local", help="Ingest files from INGEST_LOCAL_PATH into Azure AI Search"
    )
    ingest_web_parser = sub.add_parser(
        "ingest-web", help="Crawl INGEST_START_URLS and index pages + PDFs into Azure AI Search"
    )

    for parser_obj in (ingest_local_parser, ingest_web_parser):
        parser_obj.add_argument(
            "--chunker",
            choices=["doclingchunker", "simplechunker"],
            default="doclingchunker",
            help=(
                "Chunker implementation to use; defaults to doclingchunker. "
                "Current options: doclingchunker, simplechunker"
            ),
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
