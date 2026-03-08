from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import openai

from meta_rag.ingestion.chunker import Chunker
from meta_rag.ingestion.extractor import MetadataExtractor
from meta_rag.ingestion.pipeline import IngestionPipeline
from meta_rag.query.executor import ToolExecutor
from meta_rag.query.pipeline import QueryPipeline
from meta_rag.schema import MetadataField, MetadataSchema, SchemaEvolutionResult
from meta_rag.stores.base import SearchResult
from meta_rag.stores.relational import SQLiteRelationalStore
from meta_rag.stores.vector import ChromaVectorStore

__all__ = [
    "MetaRAG",
    "MetadataField",
    "MetadataSchema",
    "SchemaEvolutionResult",
    "SearchResult",
]


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
        self.last_sql: str | None = None
        self._resolve_schema(schema)

    def _resolve_schema(self, schema: list[MetadataField] | None) -> None:
        """Determine schema from DB, user input, or leave for auto-discovery."""
        db_path = os.path.join(self.data_dir, "metadata.db")
        if os.path.exists(db_path):
            existing = self.relational_store.get_schema_from_db()
            if existing and existing.fields:
                self.schema = existing
                self.vector_store.initialize(existing)
                return

        if schema:
            self.schema = MetadataSchema(fields=schema)

    def ingest(self, path: str | list[str]) -> None:
        """Ingest documents from a path or list of paths.

        Only processes new or changed documents (hash-based). If no schema is
        set, auto-discovers one from a sample of documents.
        """
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

        if self.schema is None:
            self.schema = pipeline.discover_schema(paths)

        pipeline.ingest(paths, self.schema)

    def query(self, question: str, evolve: bool = False) -> str:
        """Ask a question — meta-rag handles routing automatically.

        If evolve=True, checks for schema gaps after answering and auto-adds
        any detected missing field (with an informational message appended).
        """
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

        answer = query_pipeline.query(question)
        self.last_sql = query_pipeline.last_sql

        if evolve:
            result = self._detect_schema_gap(question)
            if result.gap_detected and result.proposed_field:
                self.add_field(result.proposed_field)
                answer = answer + result.message

        return answer

    def _detect_schema_gap(self, question: str) -> SchemaEvolutionResult:
        """Use LLM to check if the question requires a field missing from the schema."""
        current_fields = [
            f"{f.name} ({f.type}): {f.description}" for f in self.schema.fields
        ]
        fields_text = "\n".join(current_fields) if current_fields else "(no fields defined)"

        system_message = (
            "You are a schema analyst for a structured metadata database. "
            "Given a user question and the current schema, decide if the question requires "
            "a SHORT, STRUCTURED metadata field that is missing.\n\n"
            "A valid gap is a field that is:\n"
            "  - atomic and short (e.g. a name, a year, a category, a country)\n"
            "  - useful for filtering or aggregation (GROUP BY, COUNT, WHERE)\n"
            "  - not already answerable by searching the document text\n\n"
            "Do NOT flag a gap for:\n"
            "  - open-ended or descriptive questions (e.g. 'tell me about', 'explain', 'describe')\n"
            "  - fields that would store long text or summaries (e.g. biography, summary, description)\n"
            "  - questions already well-served by semantic/vector search\n"
            "  - counting or aggregation questions that are fully answerable with existing fields "
            "(e.g. 'how many people total' needs no new field; but 'how many died after 1800' "
            "DOES need year_of_death if it doesn't exist)\n"
            "  - identifier or system fields (e.g. person_id, record_id, id, key, index)\n"
            "  - fields that already exist in the schema\n"
            "  - fields that are SEMANTICALLY EQUIVALENT to an existing field, even if named "
            "differently (e.g. 'field_of_expertise', 'profession', 'role', 'specialization' "
            "are all covered by 'occupation')\n\n"
            "Return JSON with:\n"
            "- gap_detected (bool)\n"
            "- reasoning (string): brief explanation\n"
            "- proposed_field (object or null): if gap_detected, {name (snake_case), type ('text' or 'integer'), description}\n\n"
            f"Current schema fields:\n{fields_text}"
        )

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)
        gap_detected = bool(raw.get("gap_detected", False))
        reasoning = raw.get("reasoning", "")
        proposed_field: MetadataField | None = None

        if gap_detected and raw.get("proposed_field"):
            pf = raw["proposed_field"]
            proposed_field = MetadataField(
                name=pf.get("name", ""),
                type=pf.get("type", "text"),
                description=pf.get("description", ""),
            )

        if gap_detected and proposed_field:
            message = (
                f"\n\n[Schema Gap Detected] A new field '{proposed_field.name}' "
                f"({proposed_field.description}) has been added to the schema. "
                f"Run backfill() to populate it from existing documents."
            )
        else:
            message = ""

        return SchemaEvolutionResult(
            gap_detected=gap_detected,
            reasoning=reasoning,
            proposed_field=proposed_field,
            message=message,
        )

    def add_field(self, field: MetadataField) -> None:
        """Add a new field to the live schema and the database."""
        if self.schema is None:
            raise RuntimeError("No schema loaded.")
        if any(f.name == field.name for f in self.schema.fields):
            return  # already exists
        self.relational_store.add_column(field)
        self.schema.add_field(field)

    def backfill(
        self, on_progress: Callable[[int, int], None] | None = None
    ) -> dict:
        """Populate newly-added fields from already-stored chunk texts.

        For each stored chunk, runs a single LLM extraction call covering all
        unpopulated fields. After backfill, prunes fields that remained all-NULL.

        Returns:
            {"populated": [field names with at least one value],
             "pruned":    [field names that were dropped — all-NULL after backfill]}
        """
        if self.schema is None:
            return {"populated": [], "pruned": []}

        unpopulated = self.relational_store.get_unpopulated_fields(self.schema)
        if not unpopulated:
            return {"populated": [], "pruned": []}

        all_chunks = self.vector_store.get_all_chunks()
        if not all_chunks:
            pruned = self._prune_empty_fields()
            return {"populated": [], "pruned": pruned}

        backfill_schema = MetadataSchema(fields=unpopulated)
        extractor = MetadataExtractor(llm_model=self.llm_model)

        total = len(all_chunks)
        for idx, (doc_id, chunk_id, text) in enumerate(all_chunks):
            metadata = extractor.extract(text, backfill_schema)
            for field_name, value in metadata.items():
                if value is not None:
                    self.relational_store.update_metadata_field(doc_id, field_name, value)
                    self.vector_store.update_metadata(chunk_id, {field_name: value})
            if on_progress:
                on_progress(idx + 1, total)

        pruned = self._prune_empty_fields()
        populated = [f.name for f in unpopulated if f.name not in pruned]
        return {"populated": populated, "pruned": pruned}

    def _prune_empty_fields(self) -> list[str]:
        """Drop all-NULL columns from both the DB and in-memory schema."""
        empty = self.relational_store.get_empty_fields()
        for field_name in empty:
            self.relational_store.drop_column(field_name)
            self.schema.remove_field(field_name)
        return empty
