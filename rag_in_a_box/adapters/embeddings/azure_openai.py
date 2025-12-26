from __future__ import annotations

from dataclasses import dataclass
from openai import OpenAI


@dataclass
class AzureOpenAIEmbedder:
    endpoint: str
    api_key: str
    deployment: str

    def __post_init__(self) -> None:
        # Azure OpenAI: base_url ends with /openai/v1/ :contentReference[oaicite:2]{index=2}
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/openai/v1/",
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Azure OpenAI: model=deployment_name :contentReference[oaicite:3]{index=3}
        resp = self._client.embeddings.create(model=self.deployment, input=texts)
        return [d.embedding for d in resp.data]

    def embedding_dim(self) -> int:
        # Avoid guessing dimensions: ask the API once
        return len(self.embed_texts(["dim probe"])[0])
