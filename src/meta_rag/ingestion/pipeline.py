from __future__ import annotations

import hashlib
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

    def _expand_to_files(self, paths: list[str | Path]) -> list[Path]:
        """Expand a list of file/directory paths to a flat list of .txt files."""
        files: list[Path] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                files.extend(sorted(p.glob("*.txt")))
            elif p.is_file():
                files.append(p)
        return files

    def ingest(self, paths: list[str | Path], schema: MetadataSchema) -> None:
        self.vector_store.initialize(schema)
        self.relational_store.initialize(schema)

        files = self._expand_to_files(paths)

        for file_path in files:
            doc_id = file_path.stem
            content = file_path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            stored_hash = self.relational_store.get_document_hash(doc_id)
            if stored_hash == content_hash:
                continue  # unchanged — skip

            # Remove stale data for this document before re-ingesting
            self.vector_store.delete_document(doc_id)
            self.relational_store.delete_document(doc_id)

            chunks = self.chunker.chunk_file(file_path)
            for chunk in chunks:
                metadata = self.extractor.extract(chunk.text, schema)
                self.vector_store.add(chunk.doc_id, chunk.chunk_id, chunk.text, None, metadata)
                self.relational_store.insert(chunk.doc_id, metadata)

            self.relational_store.set_document_hash(doc_id, content_hash)

    def discover_schema(
        self, paths: list[str | Path], sample_count: int = 5
    ) -> MetadataSchema:
        chunks = self.chunker.chunk_paths(paths)
        sample_texts = [chunk.text for chunk in chunks[:sample_count]]
        return self.extractor.discover_schema(sample_texts)
