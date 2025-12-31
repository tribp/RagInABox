"""Lightweight Gradio chat client for the RAG pipeline."""
from __future__ import annotations

from typing import Iterable

import gradio as gr

from rag_in_a_box.adapters.embeddings.azure_openai import AzureOpenAIEmbedder
from rag_in_a_box.adapters.llm.azure_openai import AzureOpenAIChatLLM
from rag_in_a_box.adapters.vectorstores.azure_ai_search import AzureAISearchVectorStore
from rag_in_a_box.config.settings import Settings
from rag_in_a_box.core.models import ChatResponse, SearchResult
from rag_in_a_box.pipelines.chat import RAGChatEngine

import logging
logging.basicConfig(level=logging.DEBUG)

logging.debug("debug message")
logging.info("info message")



def _vector_dimension(embedder: AzureOpenAIEmbedder) -> int:
    """Best-effort way to determine embedding dimensionality for index creation."""

    if hasattr(embedder, "embedding_dim"):
        return embedder.embedding_dim()  # type: ignore[attr-defined]

    sample = embedder.embed_texts(["dimension probe"])[0]
    return len(sample)


def _format_sources(sources: Iterable[SearchResult]) -> str:
    blocks: list[str] = []
    for idx, result in enumerate(sources, start=1):
        chunk = result.chunk
        meta = chunk.metadata or {}
        title = meta.get("title") or meta.get("document_id") or "Untitled"
        source_type = meta.get("source_type") or meta.get("mime_type") or "unknown"
        snippet = chunk.text.strip()
        if len(snippet) > 400:
            snippet = f"{snippet[:400]}…"

        blocks.append(
            "\n".join(
                [
                    f"**[{idx}] {title}**",
                    f"- URI: {chunk.uri}",
                    f"- Source type: {source_type}",
                    f"- Score: {result.score:.3f}",
                    f"- Snippet: {snippet}",
                ]
            )
        )

    if not blocks:
        return "No supporting sources were retrieved."

    return "\n\n".join(blocks)


def _build_engine(settings: Settings) -> RAGChatEngine:
    embedder = AzureOpenAIEmbedder(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        deployment=settings.azure_openai_embedding_deployment,
        requests_per_minute=settings.embedding_requests_per_minute,
        max_concurrency=settings.embedding_max_concurrency,
    )

    vector_dim = _vector_dimension(embedder)
    vectorstore = AzureAISearchVectorStore(
        endpoint=settings.azure_search_endpoint,
        api_key=settings.azure_search_api_key,
        index_name=settings.azure_search_index,
        vector_dim=vector_dim,
    )
    vectorstore.ensure_index()

    llm = AzureOpenAIChatLLM(
        endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        deployment=settings.azure_openai_chat_deployment,
    )

    return RAGChatEngine(embedder=embedder, vectorstore=vectorstore, llm=llm, top_k=settings.top_k)


def create_interface(settings: Settings | None = None) -> gr.ChatInterface:
    """Construct a reusable Gradio chat interface backed by the RAG engine."""

    settings = settings or Settings()
    engine = _build_engine(settings)

    def _answer(message: str, history: list[tuple[str, str]]):
        try:
            response: ChatResponse = engine.answer(message)
        except Exception as exc:  # noqa: BLE001 - surfacing error to user
            return f"❌ Error answering question: {exc}", ""

        return response.answer, _format_sources(response.sources)

    return gr.ChatInterface(
        fn=_answer,
        additional_outputs=[gr.Markdown(label="Sources")],
        title="RAG in a Box",
        description=(
            "Chat with your indexed content. The assistant uses Azure OpenAI for generation "
            "and Azure AI Search for retrieval."
        ),
        stop_btn=True,
    )


def main() -> None:
    iface = create_interface()
    iface.launch(server_name="0.0.0.0", show_error=True)


if __name__ == "__main__":
    main()
