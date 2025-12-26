from __future__ import annotations

from rag_in_a_box.core.interfaces import Extractor
from rag_in_a_box.core.models import SourceItem


class ExtractorRegistry:
    def __init__(self, extractors: list[Extractor]):
        self.extractors = extractors

    def extract(self, item: SourceItem):
        for ex in self.extractors:
            if ex.can_handle(item.mime_type):
                return ex.extract(item)
        raise ValueError(f"No extractor found for mime_type={item.mime_type} uri={item.uri}")
