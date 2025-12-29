from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rag_in_a_box.adapters.chunking.docling_chunker import DoclingChunker
from rag_in_a_box.adapters.extractors.docling_extractor import DoclingExtractor
from rag_in_a_box.adapters.extractors.registry import ExtractorRegistry
from rag_in_a_box.core.interfaces import ContentSource, Embedder, VectorStore
from rag_in_a_box.core.models import Chunk, SourceItem
from rag_in_a_box.pipelines.ingestion import IngestionPipeline


@dataclass
class FakeParagraph:
    id: str
    text: str
    page: int


@dataclass
class FakeSection:
    id: str
    title: str
    paragraphs: list[FakeParagraph]


class FakeConversion:
    def __init__(self):
        self.metadata = {"doc_type": "pdf", "producer": "docling"}
        self.sections = [
            FakeSection(
                id="sec-1",
                title="Introduction",
                paragraphs=[
                    FakeParagraph(id="p1", text="Docling makes sense of documents.", page=1),
                    FakeParagraph(id="p2", text="It keeps structural cues intact.", page=1),
                ],
            ),
            FakeSection(
                id="sec-2",
                title="Details",
                paragraphs=[
                    FakeParagraph(id="p3", text="Metadata is preserved.", page=2),
                ],
            ),
        ]


class FakeConverter:
    def convert_bytes(self, data: bytes, mime_type: str):
        return FakeConversion()


class DummySource(ContentSource):
    def __init__(self, item: SourceItem):
        self._item = item

    def iter_items(self):
        yield self._item


class DummyEmbedder(Embedder):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]


class DummyVectorStore(VectorStore):
    def __init__(self):
        self.upserts: list[tuple[list[Chunk], list[list[float]]]] = []

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.upserts.append((chunks, vectors))

    def query(self, query_vector: list[float], k: int, filters=None):
        return []


sample_item = SourceItem(
    uri="memory://doc.pdf",
    data=b"%PDF-1.4 placeholder",
    mime_type="application/pdf",
    metadata={"source": "unit-test"},
)


def test_docling_extractor_preserves_metadata():
    extractor = DoclingExtractor(converter=FakeConverter())
    document = extractor.extract(sample_item)

    assert document.metadata["docling"]["metadata"]["doc_type"] == "pdf"
    assert len(document.metadata["docling"]["sections"]) == 2
    assert "Introduction" in document.content


def test_docling_pipeline_with_chunker():
    registry = ExtractorRegistry([DoclingExtractor(converter=FakeConverter())])
    chunker = DoclingChunker()
    embedder = DummyEmbedder()
    store = DummyVectorStore()
    pipeline = IngestionPipeline(
        source=DummySource(sample_item),
        extractor_registry=registry,
        chunker=chunker,
        embedder=embedder,
        vectorstore=store,
        batch_size=10,
    )

    stats = pipeline.run()

    assert stats.items_seen == 1
    assert stats.docs_extracted == 1
    assert stats.chunks_created == 3

    assert len(store.upserts) == 1
    chunks, vectors = store.upserts[0]
    assert len(chunks) == 3
    assert len(vectors) == 3

    first_chunk = chunks[0]
    assert first_chunk.metadata.get("section_title") == "Introduction"
    assert first_chunk.metadata.get("page_number") == 1
    assert first_chunk.metadata.get("source") == "unit-test"
    assert "docling" in first_chunk.metadata
