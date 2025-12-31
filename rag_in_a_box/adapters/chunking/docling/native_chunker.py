from rag_in_a_box.core.models import HybridDocument, Chunk

class NativeDoclingChunker:
    def chunk(self, doc: HybridDocument) -> list[Chunk]:
        if doc.is_docling:
            # Use Docling's native chunking with full contextualization
            docling_chunks = doc.get_chunks_with_context(
                max_tokens=self.max_tokens,
                overlap_tokens=self.overlap_tokens
            )
            
            # Convert to your Chunk model
            return [self._convert_docling_chunk(chunk, doc) for chunk in docling_chunks]
        
        # Fallback for non-Docling documents
        return self._fallback_chunk(doc)