from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from duo_rag.ingestion.chunker import Chunk, Chunker
from duo_rag.ingestion.extractor import MetadataExtractor
from duo_rag.ingestion.pipeline import IngestionPipeline
from duo_rag.schema import MetadataField, MetadataSchema
from duo_rag.stores.relational import SQLiteRelationalStore
from duo_rag.stores.vector import ChromaVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schema() -> MetadataSchema:
    """Return a simple two-field metadata schema used across tests."""
    return MetadataSchema(
        fields=[
            MetadataField(name="author", type="text", description="Author name"),
            MetadataField(name="year", type="integer", description="Publication year"),
        ]
    )


def _mock_openai_response(content: str) -> MagicMock:
    """Build a mock that looks like an OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


# ===========================================================================
# Chunker tests
# ===========================================================================


class TestChunkerBasicSplitting:
    """Text longer than chunk_size produces multiple chunks with correct IDs."""

    def test_produces_multiple_chunks(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "a" * 25  # well over chunk_size
        chunks = chunker.chunk_text(text, doc_id="doc1")

        assert len(chunks) > 1

    def test_chunk_ids_are_sequential(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        text = "a" * 25
        chunks = chunker.chunk_text(text, doc_id="doc1")

        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"doc1_chunk_{idx}"
            assert chunk.doc_id == "doc1"

    def test_all_text_covered(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=0)
        text = "abcdefghij" * 3  # 30 chars
        chunks = chunker.chunk_text(text, doc_id="doc1")

        reconstructed = "".join(c.text for c in chunks)
        assert reconstructed == text


class TestChunkerShortText:
    """Text shorter than chunk_size produces a single chunk."""

    def test_single_chunk_returned(self):
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        text = "short text"
        chunks = chunker.chunk_text(text, doc_id="tiny")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_id == "tiny_chunk_0"
        assert chunks[0].doc_id == "tiny"


class TestChunkerExactSize:
    """Text exactly at chunk_size produces a single chunk."""

    def test_exact_size_single_chunk(self):
        chunker = Chunker(chunk_size=20, chunk_overlap=5)
        text = "x" * 20
        chunks = chunker.chunk_text(text, doc_id="exact")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_id == "exact_chunk_0"


class TestChunkerOverlap:
    """Verify overlapping content exists between consecutive chunks."""

    def test_consecutive_chunks_share_overlap(self):
        chunker = Chunker(chunk_size=10, chunk_overlap=4)
        # Use a text length that is a multiple of step so the overlap
        # assertion holds cleanly for all consecutive pairs.
        text = "0123456789abcdef"  # 16 chars; step=6 -> chunks at 0,6,12
        chunks = chunker.chunk_text(text, doc_id="ovl")

        assert len(chunks) >= 2

        step = chunker.chunk_size - chunker.chunk_overlap
        for i in range(len(chunks) - 1):
            current_text = chunks[i].text
            next_text = chunks[i + 1].text
            # The overlap region: the last `chunk_overlap` characters of
            # the current chunk's first `step` + overlap portion should
            # appear at the start of the next chunk.
            overlap_size = min(chunker.chunk_overlap, len(next_text))
            overlap_from_current = current_text[step : step + overlap_size]
            overlap_from_next = next_text[:overlap_size]
            assert overlap_from_current == overlap_from_next, (
                f"Chunk {i} overlap '{overlap_from_current}' != "
                f"chunk {i + 1} start '{overlap_from_next}'"
            )


class TestChunkerFile:
    """chunk_file reads a tmp file and returns correct chunks."""

    def test_chunk_file(self, tmp_path):
        file = tmp_path / "sample.txt"
        file.write_text("hello world this is a test", encoding="utf-8")

        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk_file(file)

        assert len(chunks) >= 1
        assert all(c.doc_id == "sample" for c in chunks)
        assert chunks[0].chunk_id == "sample_chunk_0"


class TestChunkerDirectory:
    """chunk_directory reads multiple .txt files from a directory."""

    def test_chunk_directory(self, tmp_path):
        (tmp_path / "alpha.txt").write_text("A" * 30, encoding="utf-8")
        (tmp_path / "beta.txt").write_text("B" * 30, encoding="utf-8")
        # Non-.txt file should be ignored
        (tmp_path / "ignore.csv").write_text("C" * 30, encoding="utf-8")

        chunker = Chunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk_directory(tmp_path)

        doc_ids = {c.doc_id for c in chunks}
        assert "alpha" in doc_ids
        assert "beta" in doc_ids
        assert "ignore" not in doc_ids
        assert len(chunks) > 2  # each file produces multiple chunks


# ===========================================================================
# Extractor tests (mock OpenAI)
# ===========================================================================


class TestExtractorExtract:
    """extract() returns correct dict with type validation."""

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_extract_returns_typed_dict(self, mock_openai_cls):
        schema = _make_schema()

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": "Alice", "year": "2024"})
        )

        extractor = MetadataExtractor(llm_model="test-model")
        result = extractor.extract("some text about Alice", schema)

        assert result["author"] == "Alice"
        assert result["year"] == 2024  # integer cast
        assert isinstance(result["year"], int)

        # Verify OpenAI was called with expected args
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_extract_null_handling(self, mock_openai_cls):
        """Null values in response are preserved as None."""
        schema = _make_schema()

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": None, "year": None})
        )

        extractor = MetadataExtractor(llm_model="test-model")
        result = extractor.extract("text with no metadata", schema)

        assert result["author"] is None
        assert result["year"] is None


    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_extract_list_values_joined_as_csv(self, mock_openai_cls):
        """List values from the LLM are joined into a comma-separated string."""
        schema = MetadataSchema(
            fields=[MetadataField(name="interests", type="text", description="Interests")]
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"interests": ["astronomy", "physics"]})
        )

        extractor = MetadataExtractor(llm_model="test-model")
        result = extractor.extract("text about interests", schema)

        assert result["interests"] == "astronomy, physics"

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_extract_all_null_list_returns_none(self, mock_openai_cls):
        """A list of only null values normalises to None."""
        schema = MetadataSchema(
            fields=[MetadataField(name="interests", type="text", description="Interests")]
        )

        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"interests": [None, None]})
        )

        extractor = MetadataExtractor(llm_model="test-model")
        result = extractor.extract("text about interests", schema)

        assert result["interests"] is None


class TestExtractorDiscoverSchema:
    """discover_schema() returns a MetadataSchema with correct fields."""

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_discover_schema(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps(
                {
                    "fields": [
                        {
                            "name": "topic",
                            "type": "text",
                            "description": "Main topic",
                        },
                        {
                            "name": "page_count",
                            "type": "integer",
                            "description": "Number of pages",
                        },
                    ]
                }
            )
        )

        extractor = MetadataExtractor(llm_model="test-model")
        schema = extractor.discover_schema(["sample text one", "sample text two"])

        assert isinstance(schema, MetadataSchema)
        assert len(schema.fields) == 2
        assert schema.fields[0].name == "topic"
        assert schema.fields[0].type == "text"
        assert schema.fields[1].name == "page_count"
        assert schema.fields[1].type == "integer"
        assert schema.fields[1].description == "Number of pages"


# ===========================================================================
# Integration test
# ===========================================================================


class TestIngestionPipelineIntegration:
    """End-to-end test with mocked LLM but real SQLite and Chroma stores."""

    @patch("duo_rag.ingestion.extractor.openai.OpenAI")
    def test_ingest_populates_both_stores(self, mock_openai_cls, tmp_path):
        # --- Set up mock OpenAI client ---
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            json.dumps({"author": "TestAuthor", "year": "2025"})
        )

        # --- Prepare schema ---
        schema = _make_schema()

        # --- Create a small text file ---
        text_file = tmp_path / "testdoc.txt"
        text_file.write_text(
            "This is a test document written by TestAuthor in 2025.",
            encoding="utf-8",
        )

        # --- Initialise real stores in tmp dirs ---
        db_path = str(tmp_path / "test.db")
        chroma_dir = str(tmp_path / "chroma")

        relational_store = SQLiteRelationalStore(db_path=db_path)
        vector_store = ChromaVectorStore(persist_dir=chroma_dir)

        # --- Build and run the pipeline ---
        chunker = Chunker(chunk_size=500, chunk_overlap=50)
        extractor = MetadataExtractor(llm_model="test-model")

        pipeline = IngestionPipeline(
            chunker=chunker,
            extractor=extractor,
            vector_store=vector_store,
            relational_store=relational_store,
        )

        pipeline.ingest(paths=[str(text_file)], schema=schema)

        # --- Verify relational store ---
        rows = relational_store.execute_sql(
            "SELECT COUNT(*) AS cnt FROM documents_metadata"
        )
        assert rows[0]["cnt"] >= 1

        detail_rows = relational_store.execute_sql(
            "SELECT doc_id, author, year FROM documents_metadata"
        )
        assert len(detail_rows) >= 1
        assert detail_rows[0]["author"] == "TestAuthor"
        assert detail_rows[0]["year"] == 2025
        assert detail_rows[0]["doc_id"] == "testdoc"

        # --- Verify vector store ---
        results = vector_store.search(query_text="test document", top_k=5)
        assert len(results) >= 1
        assert results[0].doc_id == "testdoc"
        assert results[0].metadata["author"] == "TestAuthor"
