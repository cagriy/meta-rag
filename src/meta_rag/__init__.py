from __future__ import annotations

import os
from pathlib import Path

from meta_rag.ingestion.chunker import Chunker
from meta_rag.ingestion.extractor import MetadataExtractor
from meta_rag.ingestion.pipeline import IngestionPipeline
from meta_rag.query.executor import ToolExecutor
from meta_rag.query.pipeline import QueryPipeline
from meta_rag.schema import MetadataField, MetadataSchema
from meta_rag.stores.base import SearchResult
from meta_rag.stores.relational import SQLiteRelationalStore
from meta_rag.stores.vector import ChromaVectorStore

__all__ = ["MetaRAG", "MetadataField", "MetadataSchema", "SearchResult"]


class MetaRAG:
    """High-level facade for the meta-rag library.

    Combines vector search (ChromaDB) with relational queries (SQLite)
    and uses LLM tool-calling to route questions to the right backend.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o",
        schema: list[MetadataField] | None = None,
        data_dir: str = "./meta_rag_data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        vector_store: ChromaVectorStore | None = None,
        relational_store: SQLiteRelationalStore | None = None,
    ) -> None:
        self.llm_model = llm_model
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default stores
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=os.path.join(data_dir, "chroma"),
        )
        self.relational_store = relational_store or SQLiteRelationalStore(
            db_path=os.path.join(data_dir, "metadata.db"),
        )

        # Schema lifecycle:
        # 1. Try to read from existing DB
        # 2. Else use provided schema list
        # 3. Else remains None — auto-discover on first ingest()
        self.schema: MetadataSchema | None = None
        self._resolve_schema(schema)

    def _resolve_schema(self, schema: list[MetadataField] | None) -> None:
        """Determine schema from DB, user input, or leave for auto-discovery."""
        db_path = os.path.join(self.data_dir, "metadata.db")
        if os.path.exists(db_path):
            existing = self.relational_store.get_schema_from_db()
            if existing and existing.fields:
                self.schema = existing
                return

        if schema:
            self.schema = MetadataSchema(fields=schema)

    def ingest(self, path: str | list[str]) -> None:
        """Ingest documents from a path or list of paths.

        If no schema is set, auto-discovers one from a sample of documents.
        """
        # Normalize to list
        if isinstance(path, str):
            paths = [path]
        else:
            paths = list(path)

        chunker = Chunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        extractor = MetadataExtractor(llm_model=self.llm_model)
        pipeline = IngestionPipeline(
            chunker=chunker,
            extractor=extractor,
            vector_store=self.vector_store,
            relational_store=self.relational_store,
        )

        # Auto-discover schema if needed
        if self.schema is None:
            self.schema = pipeline.discover_schema(paths)

        pipeline.ingest(paths, self.schema)

    def query(self, question: str) -> str:
        """Ask a question — meta-rag handles routing automatically."""
        if self.schema is None:
            raise RuntimeError(
                "No schema available. Ingest documents first or provide a schema."
            )

        tool_executor = ToolExecutor(
            vector_store=self.vector_store,
            relational_store=self.relational_store,
        )
        query_pipeline = QueryPipeline(
            llm_model=self.llm_model,
            tool_executor=tool_executor,
            schema=self.schema,
        )

        return query_pipeline.query(question)
