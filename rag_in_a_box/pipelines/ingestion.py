from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib

from rag_in_a_box.core.interfaces import Chunker, ContentSource, Embedder, VectorStore
from rag_in_a_box.core.models import Chunk
from rag_in_a_box.adapters.extractors.registry import ExtractorRegistry


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
        batch_size: int = 16,
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

        # One timestamp for the whole ingestion run (UTC)
        run_ingested_at = datetime.now(timezone.utc).isoformat()

        def flush() -> None:
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
            except Exception:
                # keep going; later we’ll add logging
                continue

            stats.docs_extracted += 1

            chunks = self.chunker.chunk(doc)

            # Enrich chunk metadata (Chunk is frozen, so create new objects)
            enriched: list[Chunk] = []
            for c in chunks:
                md = dict(c.metadata or {})
                md["ingested_at"] = run_ingested_at
                md["content_hash"] = hashlib.sha256(c.text.encode("utf-8")).hexdigest()
                enriched.append(
                    Chunk(
                        id=c.id,
                        document_id=c.document_id,
                        uri=c.uri,
                        text=c.text,
                        metadata=md,
                    )
                )

            stats.chunks_created += len(enriched)
            pending_chunks.extend(enriched)

            if len(pending_chunks) >= self.batch_size:
                flush()

        flush()
        return stats
