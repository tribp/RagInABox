from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from openai import OpenAI


class _RateLimiter:
    def __init__(self, *, max_per_minute: Optional[int], clock: Callable[[], float] = time.monotonic):
        self.max_per_minute = max_per_minute
        self._clock = clock
        self._timestamps = deque[float]()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if not self.max_per_minute or self.max_per_minute <= 0:
            return

        window_seconds = 60.0
        while True:
            with self._lock:
                now = self._clock()
                cutoff = now - window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return

                wait_time = window_seconds - (now - self._timestamps[0])

            # Sleep outside of the lock to avoid blocking other threads from checking
            time.sleep(wait_time)


@dataclass
class AzureOpenAIEmbedder:
    endpoint: str
    api_key: str
    deployment: str
    requests_per_minute: Optional[int] = None
    max_concurrency: int = 5

    def __post_init__(self) -> None:
        # Azure OpenAI: base_url ends with /openai/v1/ :contentReference[oaicite:2]{index=2}
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=f"{self.endpoint.rstrip('/')}/openai/v1/",
        )
        self._concurrency = threading.Semaphore(self.max_concurrency)
        self._rate_limiter = _RateLimiter(max_per_minute=self.requests_per_minute)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Azure OpenAI: model=deployment_name :contentReference[oaicite:3]{index=3}
        with self._concurrency:
            self._rate_limiter.acquire()
            resp = self._client.embeddings.create(model=self.deployment, input=texts)
        return [d.embedding for d in resp.data]

    def embedding_dim(self) -> int:
        # Avoid guessing dimensions: ask the API once
        return len(self.embed_texts(["dim probe"])[0])
