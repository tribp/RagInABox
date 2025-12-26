from __future__ import annotations

import uuid

from rag_in_a_box.adapters.embeddings.dummy_embedder import DummyHashEmbedder
from rag_in_a_box.adapters.llm.dummy_llm import DummyLLM
from rag_in_a_box.adapters.vectorstores.in_memory import InMemoryVectorStore
from rag_in_a_box.core.models import Chunk
from rag_in_a_box.pipelines.chat import RAGChatEngine


def _cid() -> str:
    return str(uuid.uuid4())


def main() -> None:
    embedder = DummyHashEmbedder(dim=64)
    store = InMemoryVectorStore()
    llm = DummyLLM()

    # Pretend these are produced by extractor+chunker (we’ll build those later)
    chunks = [
        Chunk(id=_cid(), document_id="doc1", uri="local://doc1", text="RAG combines retrieval with generation."),
        Chunk(id=_cid(), document_id="doc1", uri="local://doc1", text="Azure AI Search can be used as a vector database."),
        Chunk(id=_cid(), document_id="doc2", uri="https://example.com", text="Gradio can host a chat UI for LLM apps."),
    ]
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.upsert(chunks, vectors)

    engine = RAGChatEngine(embedder=embedder, vectorstore=store, llm=llm, top_k=2)

    q = "What can be used as the vector database?"
    resp = engine.answer(q)

    print("QUESTION:", q)
    print("\nANSWER:\n", resp.answer)
    print("\nSOURCES:")
    for r in resp.sources:
        print(f"- score={r.score:.3f} uri={r.chunk.uri} text={r.chunk.text}")


if __name__ == "__main__":
    main()
