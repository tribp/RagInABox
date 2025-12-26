from __future__ import annotations

import uuid

from rag_in_a_box.config.settings import Settings
from rag_in_a_box.adapters.embeddings.azure_openai import AzureOpenAIEmbedder
from rag_in_a_box.adapters.llm.azure_openai import AzureOpenAIChatLLM
from rag_in_a_box.adapters.vectorstores.azure_ai_search import AzureAISearchVectorStore
from rag_in_a_box.core.models import Chunk
from rag_in_a_box.pipelines.chat import RAGChatEngine


def _id() -> str:
    return str(uuid.uuid4())


def main() -> None:
    s = Settings()

    embedder = AzureOpenAIEmbedder(
        endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        deployment=s.azure_openai_embedding_deployment,
    )

    vector_dim = embedder.embedding_dim()  # don’t guess dimensions
    store = AzureAISearchVectorStore(
        endpoint=s.azure_search_endpoint,
        api_key=s.azure_search_api_key,
        index_name=s.azure_search_index,
        vector_dim=vector_dim,
    )
    store.ensure_index()

    llm = AzureOpenAIChatLLM(
        endpoint=s.azure_openai_endpoint,
        api_key=s.azure_openai_api_key,
        deployment=s.azure_openai_chat_deployment,
    )

    # Index a few chunks
    chunks = [
        Chunk(id=_id(), document_id="demo", uri="local://demo", text="Azure AI Search can be used as a vector database.", metadata={"chunk_index": 0}),
        Chunk(id=_id(), document_id="demo", uri="local://demo", text="Gradio can host a chat UI for LLM apps.", metadata={"chunk_index": 1}),
    ]
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.upsert(chunks, vectors)

    # Ask a question
    engine = RAGChatEngine(embedder=embedder, vectorstore=store, llm=llm, top_k=s.top_k)
    resp = engine.answer("What can be used as the vector database?")

    print(resp.answer)
    print("\nSOURCES:")
    for r in resp.sources:
        print(f"- score={r.score:.3f} uri={r.chunk.uri} text={r.chunk.text}")


if __name__ == "__main__":
    main()
