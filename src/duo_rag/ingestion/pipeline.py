from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from duo_rag.ingestion.chunker import Chunker
from duo_rag.ingestion.extractor import MetadataExtractor
from duo_rag.schema import MetadataSchema
from duo_rag.stores.base import RelationalStore, VectorStore


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

    def ingest(
        self,
        paths: list[str | Path],
        schema: MetadataSchema,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> dict:
        self.vector_store.initialize(schema)
        self.relational_store.initialize(schema)

        files = self._expand_to_files(paths)
        total = len(files)
        unchanged = 0
        changed = 0
        new = 0

        for idx, file_path in enumerate(files):
            doc_id = file_path.stem
            content = file_path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            stored_hash = self.relational_store.get_document_hash(doc_id)
            if stored_hash == content_hash:
                unchanged += 1
            else:
                if stored_hash is None:
                    new += 1
                else:
                    changed += 1

                # Remove stale data for this document before re-ingesting
                self.vector_store.delete_document(doc_id)
                self.relational_store.delete_document(doc_id)

                chunks = self.chunker.chunk_file(file_path)
                metadata = self.extractor.extract(content, schema)
                for chunk in chunks:
                    chunk_metadata = {
                        **metadata,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                    }
                    self.vector_store.add(chunk.doc_id, chunk.chunk_id, chunk.text, None, chunk_metadata)
                self.relational_store.insert(doc_id, metadata)

                self.relational_store.set_document_hash(doc_id, content_hash)

            if on_progress:
                on_progress(idx + 1, total)

        return {"unchanged": unchanged, "changed": changed, "new": new}

    def discover_schema(
        self,
        paths: list[str | Path],
        sample_power: float = 0.5,
        max_samples: int = 50,
    ) -> MetadataSchema:
        files = self._expand_to_files(paths)
        if not files:
            raise ValueError("No documents found for schema discovery")

        sample_count = max(5, min(max_samples, round(len(files) ** sample_power)))
        step = max(1, len(files) // sample_count)
        sampled_files = files[::step][:sample_count]

        sample_texts = [Path(f).read_text(encoding="utf-8") for f in sampled_files]
        return self.extractor.discover_schema(sample_texts)
