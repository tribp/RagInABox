from __future__ import annotations

from dataclasses import dataclass
from openai import OpenAI

from rag_in_a_box.core.models import ChatMessage


@dataclass
class AzureOpenAIChatLLM:
    endpoint: str
    api_key: str
    deployment: str

    def __post_init__(self) -> None:
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/openai/v1/",
        )

    def chat(self, messages: list[ChatMessage], **kwargs) -> str:
        # Azure OpenAI: model=deployment_name :contentReference[oaicite:5]{index=5}
        resp = self._client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=kwargs.get("temperature", 0.2),
        )
        return resp.choices[0].message.content or ""
