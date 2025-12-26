from __future__ import annotations

from dataclasses import dataclass

from rag_in_a_box.core.interfaces import Chunker, Embedder, VectorStore
from rag_in_a_box.core.models import Chunk
from rag_in_a_box.adapters.extractors.registry import ExtractorRegistry
from rag_in_a_box.core.interfaces import ContentSource


@dataclass
class IngestionStats:
    items_seen: int = 0
    docs_extracted: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0


class IngestionPipeline:
    def __init__(
        self,
        *,
        source: ContentSource,
        extractor_registry: ExtractorRegistry,
        chunker: Chunker,
        embedder: Embedder,
        vectorstore: VectorStore,
        batch_size: int = 64,
    ):
        self.source = source
        self.extractor_registry = extractor_registry
        self.chunker = chunker
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.batch_size = batch_size

    def run(self) -> IngestionStats:
        stats = IngestionStats()

        pending_chunks: list[Chunk] = []

        def flush():
            nonlocal pending_chunks, stats
            if not pending_chunks:
                return
            vectors = self.embedder.embed_texts([c.text for c in pending_chunks])
            self.vectorstore.upsert(pending_chunks, vectors)
            stats.chunks_indexed += len(pending_chunks)
            pending_chunks = []

        for item in self.source.iter_items():
            stats.items_seen += 1

            try:
                doc = self.extractor_registry.extract(item)
            except Exception as e:
                # keep going; later we’ll add logging
                continue

            stats.docs_extracted += 1

            chunks = self.chunker.chunk(doc)
            stats.chunks_created += len(chunks)

            pending_chunks.extend(chunks)

            if len(pending_chunks) >= self.batch_size:
                flush()

        flush()
        return stats
