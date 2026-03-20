"""Tests for schema evolution: gap detection, add/drop columns, backfill, incremental ingest."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from duo_rag.schema import MetadataField, MetadataSchema, SchemaEvolutionResult
from duo_rag.stores.relational import SQLiteRelationalStore
from duo_rag.stores.vector import ChromaVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schema() -> MetadataSchema:
    return MetadataSchema(
        fields=[
            MetadataField(name="author", type="text", description="Author name"),
            MetadataField(name="year", type="integer", description="Publication year"),
        ]
    )


def _init_store(tmp_path: Path) -> SQLiteRelationalStore:
    store = SQLiteRelationalStore(db_path=str(tmp_path / "test.db"))
    store.initialize(_make_schema())
    return store


def _mock_openai_response(content: str) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ===========================================================================
# 1. _schema_fields persistence
# ===========================================================================


class TestSchemaFieldsPersistence:
    def test_descriptions_survive_round_trip(self, tmp_path):
        store = _init_store(tmp_path)
        schema2 = store.get_schema_from_db()

        assert schema2 is not None
        author = next(f for f in schema2.fields if f.name == "author")
        assert author.description == "Author name"
        assert author.type == "text"

        year = next(f for f in schema2.fields if f.name == "year")
        assert year.description == "Publication year"
        assert year.type == "integer"

    def test_initialize_creates_schema_fields_table(self, tmp_path):
        store = _init_store(tmp_path)
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_fields'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_initialize_is_idempotent(self, tmp_path):
        store = _init_store(tmp_path)
        store.initialize(_make_schema())  # second call — should not raise
        schema = store.get_schema_from_db()
        assert len(schema.fields) == 2


# ===========================================================================
# 2. add_column
# ===========================================================================


class TestAddColumn:
    def test_column_added_to_documents_metadata(self, tmp_path):
        store = _init_store(tmp_path)
        new_field = MetadataField(name="country", type="text", description="Country of origin")
        store.add_column(new_field)

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute("PRAGMA table_info(documents_metadata)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "country" in cols

    def test_column_added_to_schema_fields_table(self, tmp_path):
        store = _init_store(tmp_path)
        new_field = MetadataField(name="country", type="text", description="Country of origin")
        store.add_column(new_field)

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name, type, description FROM _schema_fields WHERE name = 'country'"
        )
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[1] == "text"
        assert row[2] == "Country of origin"

    def test_added_field_retrieved_by_get_schema_from_db(self, tmp_path):
        store = _init_store(tmp_path)
        store.add_column(MetadataField(name="country", type="text", description="Country"))
        schema = store.get_schema_from_db()
        names = [f.name for f in schema.fields]
        assert "country" in names


# ===========================================================================
# 3. drop_column
# ===========================================================================


class TestDropColumn:
    def test_column_removed_from_documents_metadata(self, tmp_path):
        store = _init_store(tmp_path)
        store.add_column(MetadataField(name="country", type="text", description="Country"))
        store.drop_column("country")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute("PRAGMA table_info(documents_metadata)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "country" not in cols

    def test_column_removed_from_schema_fields_table(self, tmp_path):
        store = _init_store(tmp_path)
        store.add_column(MetadataField(name="country", type="text", description="Country"))
        store.drop_column("country")

        conn = sqlite3.connect(str(tmp_path / "test.db"))
        cursor = conn.execute(
            "SELECT name FROM _schema_fields WHERE name = 'country'"
        )
        assert cursor.fetchone() is None
        conn.close()


# ===========================================================================
# 4. get_empty_fields
# ===========================================================================


class TestGetEmptyFields:
    def test_all_null_column_is_returned(self, tmp_path):
        store = _init_store(tmp_path)
        # Insert a row with NULL for both author and year
        store.insert("doc1", {"author": None, "year": None})
        empty = store.get_empty_fields()
        assert "author" in empty
        assert "year" in empty

    def test_populated_column_not_returned(self, tmp_path):
        store = _init_store(tmp_path)
        store.insert("doc1", {"author": "Alice", "year": None})
        empty = store.get_empty_fields()
        assert "author" not in empty
        assert "year" in empty

    def test_system_columns_excluded(self, tmp_path):
        store = _init_store(tmp_path)
        empty = store.get_empty_fields()
        assert "id" not in empty
        assert "doc_id" not in empty
        assert "chunk_id" not in empty  # chunk_id no longer exists as a column


# ===========================================================================
# 5. Document hashes
# ===========================================================================


class TestDocumentHashes:
    def test_get_returns_none_for_unknown_doc(self, tmp_path):
        store = _init_store(tmp_path)
        assert store.get_document_hash("missing_doc") is None

    def test_set_then_get_returns_hash(self, tmp_path):
        store = _init_store(tmp_path)
        store.set_document_hash("doc1", "abc123")
        assert store.get_document_hash("doc1") == "abc123"

    def test_set_overwrites_existing_hash(self, tmp_path):
        store = _init_store(tmp_path)
        store.set_document_hash("doc1", "abc123")
        store.set_document_hash("doc1", "xyz789")
        assert store.get_document_hash("doc1") == "xyz789"

    def test_delete_document_removes_hash(self, tmp_path):
        store = _init_store(tmp_path)
        store.insert("doc1", {"author": "Alice", "year": 2023})
        store.set_document_hash("doc1", "abc123")
        store.delete_document("doc1")
        assert store.get_document_hash("doc1") is None

    def test_delete_document_removes_metadata_rows(self, tmp_path):
        store = _init_store(tmp_path)
        store.insert("doc1", {"author": "Alice", "year": 2023})
        store.delete_document("doc1")
        rows = store.execute_sql("SELECT COUNT(*) AS cnt FROM documents_metadata")
        assert rows[0]["cnt"] == 0

    def test_clear_all_data_removes_everything(self, tmp_path):
        store = _init_store(tmp_path)
        store.insert("doc1", {"author": "Alice", "year": 2023})
        store.set_document_hash("doc1", "abc123")
        store.clear_all_data()

        rows = store.execute_sql("SELECT COUNT(*) AS cnt FROM documents_metadata")
        assert rows[0]["cnt"] == 0
        assert store.get_document_hash("doc1") is None


# ===========================================================================
# 6. MetadataSchema.add_field / remove_field
# ===========================================================================


class TestSchemaAddRemoveField:
    def test_add_field_appends_to_fields(self):
        schema = _make_schema()
        schema.add_field(MetadataField(name="country", type="text", description="Country"))
        assert len(schema.fields) == 3
        assert schema.fields[-1].name == "country"

    def test_remove_field_removes_by_name(self):
        schema = _make_schema()
        schema.remove_field("author")
        names = [f.name for f in schema.fields]
        assert "author" not in names
        assert "year" in names

    def test_remove_nonexistent_field_is_noop(self):
        schema = _make_schema()
        schema.remove_field("nonexistent")
        assert len(schema.fields) == 2


# ===========================================================================
# 7. query() with evolve=True / evolve=False
# ===========================================================================


class TestQueryWithEvolve:
    @patch("duo_rag.__init__.openai.OpenAI")
    @patch("duo_rag.query.pipeline.openai.OpenAI")
    def test_gap_detected_appends_message_and_adds_field(
        self, mock_query_openai_cls, mock_evolve_openai_cls, tmp_path
    ):
        from duo_rag import DuoRAG

        # Set up relational store with schema
        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        mock_vector = MagicMock()
        mock_vector.search.return_value = []

        # Query pipeline: direct answer
        mock_query_client = MagicMock()
        mock_query_openai_cls.return_value = mock_query_client
        direct_response = MagicMock()
        direct_response.choices[0].finish_reason = "stop"
        direct_response.choices[0].message.content = "I cannot find that information."
        direct_response.choices[0].message.tool_calls = None
        mock_query_client.chat.completions.create.return_value = direct_response

        # Evolve: gap detected
        mock_evolve_client = MagicMock()
        mock_evolve_openai_cls.return_value = mock_evolve_client
        gap_response = _mock_openai_response(
            json.dumps({
                "gap_detected": True,
                "reasoning": "The field research_interests does not exist.",
                "proposed_field": {
                    "name": "research_interests",
                    "type": "text",
                    "description": "Main research interests",
                },
                "message": "The system has identified new information it can learn from your documents. More precise answers will be available soon.",
                "unavailable_message": "This question can't be answered precisely right now because the required data hasn't been indexed yet. Please check back later for more precise results.",
            })
        )
        mock_evolve_client.chat.completions.create.return_value = gap_response

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )

        answer = rag.query("What are the research interests?", evolve=True)

        assert "check back later" in answer
        field_names = [f.name for f in rag.schema.fields]
        assert "research_interests" in field_names


class TestQueryWithoutEvolve:
    @patch("duo_rag.query.pipeline.openai.OpenAI")
    def test_no_evolve_does_not_call_gap_detection(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        mock_vector = MagicMock()
        mock_vector.search.return_value = []

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        direct_response = MagicMock()
        direct_response.choices[0].finish_reason = "stop"
        direct_response.choices[0].message.content = "Alice."
        direct_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create.return_value = direct_response

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )

        answer = rag.query("Who is the author?", evolve=False)

        assert answer == "Alice."
        # Only 1 LLM call (the query itself) — no gap detection call
        assert mock_client.chat.completions.create.call_count == 1


# ===========================================================================
# 8. _detect_schema_gap
# ===========================================================================


class TestDetectSchemaGap:
    @patch("duo_rag.__init__.openai.OpenAI")
    def test_gap_detected(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        relational_store.initialize(_make_schema())

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "gap_detected": True,
                "reasoning": "research_interests is not in the schema",
                "proposed_field": {
                    "name": "research_interests",
                    "type": "text",
                    "description": "Main research areas",
                },
                "message": "The system has identified new information it can learn from your documents. More precise answers will be available soon.",
                "unavailable_message": "This question can't be answered precisely right now because the required data hasn't been indexed yet. Please check back later.",
            })
        )

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        result = rag._detect_schema_gap("What are the research interests?")

        assert result.gap_detected is True
        assert result.proposed_field is not None
        assert result.proposed_field.name == "research_interests"
        assert result.message != ""
        assert result.unavailable_message != ""

    @patch("duo_rag.__init__.openai.OpenAI")
    def test_no_gap_detected(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        relational_store.initialize(_make_schema())

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({
                "gap_detected": False,
                "reasoning": "author field already exists",
                "proposed_field": None,
                "message": "",
                "unavailable_message": "",
            })
        )

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        result = rag._detect_schema_gap("Who is the author?")

        assert result.gap_detected is False
        assert result.proposed_field is None
        assert result.message == ""
        assert result.unavailable_message == ""


# ===========================================================================
# 9. backfill
# ===========================================================================


class TestBackfill:
    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_backfill_populates_new_field(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        # Add a new column with no values
        new_field = MetadataField(
            name="country", type="text", description="Country of origin"
        )
        relational_store.add_column(new_field)

        # Mock vector store: one stored chunk
        mock_vector = MagicMock()
        mock_vector.get_all_chunks.return_value = [("doc1", "chunk1", "Alice from Germany")]

        # Mock LLM extraction: returns country
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"country": "Germany"})
        )

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )
        # Sync in-memory schema with the new column
        rag.schema.add_field(new_field)

        result = rag.backfill()

        assert "country" in result["populated"]
        assert "country" not in result["pruned"]

        # Verify the DB was updated
        rows = relational_store.execute_sql(
            "SELECT country FROM documents_metadata WHERE doc_id = 'doc1'"
        )
        assert rows[0]["country"] == "Germany"

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_backfill_prunes_all_null_field(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        new_field = MetadataField(
            name="country", type="text", description="Country"
        )
        relational_store.add_column(new_field)

        mock_vector = MagicMock()
        mock_vector.get_all_chunks.return_value = [("doc1", "chunk1", "Some text")]

        # LLM returns null for country
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"country": None})
        )

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )
        rag.schema.add_field(new_field)

        result = rag.backfill()

        assert "country" in result["pruned"]
        assert "country" not in result["populated"]
        # Field should be removed from in-memory schema
        assert not any(f.name == "country" for f in rag.schema.fields)

    def test_backfill_returns_early_when_no_unpopulated(self, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        mock_vector = MagicMock()

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )

        result = rag.backfill()

        assert result == {"populated": [], "pruned": []}
        mock_vector.get_all_chunks.assert_not_called()

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_backfill_calls_on_progress(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": None, "year": None})

        mock_vector = MagicMock()
        mock_vector.get_all_chunks.return_value = [
            ("doc1", "chunk1", "text1"),
            ("doc1", "chunk2", "text2"),
        ]
        # Insert second chunk so it's in the DB too
        relational_store.insert("doc1", {"author": None, "year": None})

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": "Alice", "year": 2023})
        )

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=mock_vector,
            relational_store=relational_store,
        )

        progress_calls = []
        rag.backfill(on_progress=lambda cur, tot: progress_calls.append((cur, tot)))

        assert (1, 2) in progress_calls
        assert (2, 2) in progress_calls


# ===========================================================================
# 10. Incremental ingest
# ===========================================================================


class TestIncrementalIngest:
    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_unchanged_doc_is_skipped(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        text_file = tmp_path / "doc.txt"
        text_file.write_text("Hello world", encoding="utf-8")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": "Alice", "year": 2023})
        )

        chroma_dir = str(tmp_path / "chroma")
        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        vector_store = ChromaVectorStore(persist_dir=chroma_dir)

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            schema=[
                MetadataField(name="author", type="text", description="Author"),
                MetadataField(name="year", type="integer", description="Year"),
            ],
            data_dir=str(tmp_path),
            relational_store=relational_store,
            vector_store=vector_store,
        )

        rag.ingest(str(tmp_path))
        first_call_count = mock_client.chat.completions.create.call_count

        # Second ingest: file unchanged — no new LLM calls
        rag.ingest(str(tmp_path))
        assert mock_client.chat.completions.create.call_count == first_call_count

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_changed_doc_is_re_ingested(self, mock_openai_cls, tmp_path):
        from duo_rag import DuoRAG

        text_file = tmp_path / "doc.txt"
        text_file.write_text("Hello world", encoding="utf-8")

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": "Alice", "year": 2023})
        )

        chroma_dir = str(tmp_path / "chroma")
        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        vector_store = ChromaVectorStore(persist_dir=chroma_dir)

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            schema=[
                MetadataField(name="author", type="text", description="Author"),
                MetadataField(name="year", type="integer", description="Year"),
            ],
            data_dir=str(tmp_path),
            relational_store=relational_store,
            vector_store=vector_store,
        )

        rag.ingest(str(tmp_path))
        first_call_count = mock_client.chat.completions.create.call_count

        # Modify the file
        text_file.write_text("Hello world — updated content", encoding="utf-8")

        rag.ingest(str(tmp_path))
        # Should have made at least one new extraction call
        assert mock_client.chat.completions.create.call_count > first_call_count


# ===========================================================================
# 11. _prune_empty_fields
# ===========================================================================


class TestPruneEmptyFields:
    def test_empty_field_is_pruned_from_db_and_schema(self, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        # Insert a row with NULL for both fields
        relational_store.insert("doc1", {"author": None, "year": None})

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        pruned = rag._prune_empty_fields()

        assert "author" in pruned
        assert "year" in pruned
        # Schema should have no fields left
        assert len(rag.schema.fields) == 0

    def test_populated_field_not_pruned(self, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)
        relational_store.insert("doc1", {"author": "Alice", "year": None})

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        pruned = rag._prune_empty_fields()

        assert "author" not in pruned
        assert "year" in pruned
        field_names = [f.name for f in rag.schema.fields]
        assert "author" in field_names
        assert "year" not in field_names


# ===========================================================================
# 12. add_field method on DuoRAG
# ===========================================================================


class TestAddFieldMethod:
    def test_add_field_updates_store_and_schema(self, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        new_field = MetadataField(name="country", type="text", description="Country")
        rag.add_field(new_field)

        # In-memory schema updated
        assert any(f.name == "country" for f in rag.schema.fields)

        # DB column added
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA table_info(documents_metadata)")
        cols = {row[1] for row in cursor.fetchall()}
        conn.close()
        assert "country" in cols

    def test_add_field_skips_duplicate(self, tmp_path):
        from duo_rag import DuoRAG

        db_path = str(tmp_path / "metadata.db")
        relational_store = SQLiteRelationalStore(db_path=db_path)
        schema = _make_schema()
        relational_store.initialize(schema)

        rag = DuoRAG(
            llm_model="gpt-5-mini",
            data_dir=str(tmp_path),
            vector_store=MagicMock(),
            relational_store=relational_store,
        )

        original_count = len(rag.schema.fields)
        rag.add_field(MetadataField(name="author", type="text", description="duplicate"))
        assert len(rag.schema.fields) == original_count
