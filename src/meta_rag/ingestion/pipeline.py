from __future__ import annotations

from pathlib import Path

from meta_rag.ingestion.chunker import Chunker
from meta_rag.ingestion.extractor import MetadataExtractor
from meta_rag.schema import MetadataSchema
from meta_rag.stores.base import RelationalStore, VectorStore


class IngestionPipeline:
    def __init__(
        self,
        chunker: Chunker,
        extractor: MetadataExtractor,
        vector_store: VectorStore,
        relational_store: RelationalStore,
    ) -> None:
        self.chunker = chunker
        self.extractor = extractor
        self.vector_store = vector_store
        self.relational_store = relational_store

    def ingest(self, paths: list[str | Path], schema: MetadataSchema) -> None:
        self.vector_store.initialize(schema)
        self.relational_store.initialize(schema)

        chunks = self.chunker.chunk_paths(paths)

        for chunk in chunks:
            metadata = self.extractor.extract(chunk.text, schema)
            self.vector_store.add(
                chunk.doc_id, chunk.chunk_id, chunk.text, None, metadata
            )
            self.relational_store.insert(chunk.doc_id, chunk.chunk_id, metadata)

    def discover_schema(
        self, paths: list[str | Path], sample_count: int = 5
    ) -> MetadataSchema:
        chunks = self.chunker.chunk_paths(paths)
        sample_texts = [chunk.text for chunk in chunks[:sample_count]]
        return self.extractor.discover_schema(sample_texts)
