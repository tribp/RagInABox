from __future__ import annotations

from rag_in_a_box.core.interfaces import Embedder, LLM, VectorStore
from rag_in_a_box.core.models import ChatMessage, ChatResponse


class RAGChatEngine:
    def __init__(self, *, embedder: Embedder, vectorstore: VectorStore, llm: LLM, top_k: int = 5):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.top_k = top_k

    def answer(self, question: str) -> ChatResponse:
        # 1) embed the query
        qvec = self.embedder.embed_texts([question])[0]

        # 2) retrieve
        results = self.vectorstore.query(qvec, k=self.top_k)

        # 3) build context
        context_blocks = []
        for i, r in enumerate(results, start=1):
            uri = r.chunk.uri
            context_blocks.append(f"[{i}] {uri}\n{r.chunk.text}")

        context = "\n\n".join(context_blocks) if context_blocks else "(no context found)"

        # 4) prompt + generate
        messages = [
            ChatMessage(role="system", content="You are a helpful RAG assistant. Use provided context when relevant."),
            ChatMessage(
                role="user",
                content=(
                    "Answer the question using the context.\n\n"
                    f"CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{question}"
                ),
            ),
        ]

        answer = self.llm.chat(messages)
        return ChatResponse(answer=answer, sources=results)
