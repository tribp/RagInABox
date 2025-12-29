from __future__ import annotations

import re


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str, *, model: str = "text-embedding-3-small") -> int:
    """Return a lightweight token count for ``text``.

    The regex approximates tokenization by splitting on word boundaries and
    punctuation, avoiding heavy external dependencies while providing a
    consistent, deterministic count for metadata and observability.
    """

    if not text:
        return 0

    return len(_TOKEN_PATTERN.findall(text))
