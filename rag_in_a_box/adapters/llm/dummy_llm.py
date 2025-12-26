from __future__ import annotations

from rag_in_a_box.core.models import ChatMessage


class DummyLLM:
    """
    Fake LLM for smoke testing.
    It just echoes the question and shows that it received context.
    """

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return (
            "DUMMY ANSWER\n"
            "------------\n"
            "I received your prompt. Here is the final user message:\n\n"
            f"{user_msg}\n"
        )
