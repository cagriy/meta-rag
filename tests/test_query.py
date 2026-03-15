"""Tests for the query module: ToolBuilder, SQL safety, ToolExecutor, QueryPipeline, and MetaRAG schema lifecycle."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from meta_rag.query.executor import ToolExecutor
from meta_rag.query.pipeline import QueryPipeline
from meta_rag.query.tools import ToolBuilder
from meta_rag.schema import MetadataField, MetadataSchema
from meta_rag.stores.base import SearchResult
from meta_rag.stores.relational import SQLiteRelationalStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_schema() -> MetadataSchema:
    return MetadataSchema(
        fields=[
            MetadataField(name="author", type="text", description="Author name"),
            MetadataField(name="year", type="integer", description="Publication year"),
        ]
    )


@pytest.fixture
def sqlite_store(tmp_path) -> SQLiteRelationalStore:
    """Create a real SQLite store with schema and sample data."""
    db_path = str(tmp_path / "test.db")
    store = SQLiteRelationalStore(db_path=db_path)
    schema = MetadataSchema(
        fields=[
            MetadataField(name="author", type="text", description="Author name"),
            MetadataField(name="year", type="integer", description="Publication year"),
        ]
    )
    store.initialize(schema)
    store.insert("doc1", {"author": "Alice", "year": 2023})
    store.insert("doc2", {"author": "Bob", "year": 2024})
    return store


# ---------------------------------------------------------------------------
# ToolBuilder tests
# ---------------------------------------------------------------------------


class TestToolBuilder:
    def test_build_tools_returns_list_of_two_dicts(self, sample_schema: MetadataSchema):
        tools = ToolBuilder.build_tools(sample_schema)
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert all(isinstance(t, dict) for t in tools)

    def test_build_tools_matches_openai_tool_format(self, sample_schema: MetadataSchema):
        tools = ToolBuilder.build_tools(sample_schema)

        for tool in tools:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            assert func["parameters"]["type"] == "object"
            assert "properties" in func["parameters"]
            assert "required" in func["parameters"]

        names = {t["function"]["name"] for t in tools}
        assert names == {"semantic_search", "run_sql"}


# ---------------------------------------------------------------------------
# SQL safety tests (real SQLiteRelationalStore)
# ---------------------------------------------------------------------------


class TestSQLSafety:
    def test_reject_drop_table(self, sqlite_store: SQLiteRelationalStore):
        # DROP doesn't start with SELECT, so hits the first guard
        with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
            sqlite_store.execute_sql("DROP TABLE documents_metadata;")

    def test_reject_drop_embedded_in_select(self, sqlite_store: SQLiteRelationalStore):
        # SELECT that contains DROP hits the blacklist
        with pytest.raises(ValueError, match="Forbidden SQL keyword detected: DROP"):
            sqlite_store.execute_sql(
                "SELECT * FROM documents_metadata; DROP TABLE documents_metadata;"
            )

    def test_reject_insert(self, sqlite_store: SQLiteRelationalStore):
        with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
            sqlite_store.execute_sql(
                "INSERT INTO documents_metadata (doc_id, chunk_id) VALUES ('x', 'y');"
            )

    def test_reject_update(self, sqlite_store: SQLiteRelationalStore):
        with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
            sqlite_store.execute_sql(
                "UPDATE documents_metadata SET author='Evil' WHERE doc_id='doc1';"
            )

    def test_reject_delete(self, sqlite_store: SQLiteRelationalStore):
        with pytest.raises(ValueError, match="Only SELECT statements are allowed"):
            sqlite_store.execute_sql(
                "DELETE FROM documents_metadata WHERE doc_id='doc1';"
            )

    def test_allow_valid_select(self, sqlite_store: SQLiteRelationalStore):
        rows = sqlite_store.execute_sql(
            "SELECT author, year FROM documents_metadata ORDER BY year;"
        )
        assert len(rows) == 2
        assert rows[0] == {"author": "Alice", "year": 2023}
        assert rows[1] == {"author": "Bob", "year": 2024}


# ---------------------------------------------------------------------------
# ToolExecutor tests (mock stores)
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def test_semantic_search_routes_to_vector_store(self):
        mock_vector = MagicMock()
        mock_vector.search.return_value = [
            SearchResult(
                doc_id="doc1",
                chunk_id="chunk1",
                text="Some relevant text",
                metadata={"author": "Alice"},
                score=0.95,
            )
        ]
        mock_relational = MagicMock()

        executor = ToolExecutor(
            vector_store=mock_vector, relational_store=mock_relational
        )
        result = executor.execute("semantic_search", {"query": "test"})

        mock_vector.search.assert_called_once_with(
            query_text="test", top_k=10, filters=None
        )
        assert "chunk1" in result
        assert "0.9500" in result

    def test_run_sql_routes_to_relational_store(self):
        mock_vector = MagicMock()
        mock_relational = MagicMock()
        mock_relational.execute_sql.return_value = [
            {"author": "Alice", "year": 2023}
        ]

        executor = ToolExecutor(
            vector_store=mock_vector, relational_store=mock_relational
        )
        result = executor.execute("run_sql", {"sql": "SELECT * FROM documents_metadata;"})

        mock_relational.execute_sql.assert_called_once_with(
            "SELECT * FROM documents_metadata;"
        )
        parsed = json.loads(result)
        assert parsed == [{"author": "Alice", "year": 2023}]

    def test_unknown_tool_raises_value_error(self):
        executor = ToolExecutor(
            vector_store=MagicMock(), relational_store=MagicMock()
        )
        with pytest.raises(ValueError, match="Unknown tool: nonexistent"):
            executor.execute("nonexistent", {})


# ---------------------------------------------------------------------------
# Helpers for mocking OpenAI responses
# ---------------------------------------------------------------------------


def _make_tool_call(tool_call_id: str, name: str, arguments: dict) -> MagicMock:
    """Create a mock object that mimics an OpenAI tool call."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _make_openai_response(
    finish_reason: str,
    content: str | None = None,
    tool_calls: list | None = None,
) -> MagicMock:
    """Create a mock object mimicking an OpenAI chat completion response."""
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# QueryPipeline tests (mock OpenAI)
# ---------------------------------------------------------------------------


class TestQueryPipeline:
    @patch("meta_rag.query.pipeline.openai.OpenAI")
    def test_tool_calling_flow(self, mock_openai_cls, sample_schema: MetadataSchema):
        """When the LLM decides to use a tool, the pipeline should execute it and make a second call."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: LLM returns a tool call
        tool_call = _make_tool_call(
            tool_call_id="call_123",
            name="run_sql",
            arguments={"sql": "SELECT COUNT(*) AS cnt FROM documents_metadata;"},
        )
        first_response = _make_openai_response(
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

        # Second call: LLM synthesizes the answer
        second_response = _make_openai_response(
            finish_reason="stop",
            content="There are 42 documents in the collection.",
        )

        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]

        mock_executor = MagicMock()
        mock_executor.execute.return_value = json.dumps([{"cnt": 42}])

        pipeline = QueryPipeline(
            llm_model="gpt-4o",
            tool_executor=mock_executor,
            schema=sample_schema,
        )

        answer = pipeline.query("How many documents are there?")

        assert answer == "There are 42 documents in the collection."
        assert mock_client.chat.completions.create.call_count == 2
        mock_executor.execute.assert_called_once_with(
            "run_sql",
            {"sql": "SELECT COUNT(*) AS cnt FROM documents_metadata;"},
        )

    @patch("meta_rag.query.pipeline.openai.OpenAI")
    def test_direct_answer_flow(self, mock_openai_cls, sample_schema: MetadataSchema):
        """When the LLM answers directly (no tools), the pipeline returns immediately."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        direct_response = _make_openai_response(
            finish_reason="stop",
            content="I can help you with questions about the document collection.",
        )

        mock_client.chat.completions.create.return_value = direct_response

        mock_executor = MagicMock()

        pipeline = QueryPipeline(
            llm_model="gpt-4o",
            tool_executor=mock_executor,
            schema=sample_schema,
        )

        answer = pipeline.query("Hello!")

        assert answer == "I can help you with questions about the document collection."
        assert mock_client.chat.completions.create.call_count == 1
        mock_executor.execute.assert_not_called()

    @patch("meta_rag.query.pipeline.openai.OpenAI")
    def test_fallback_false_blocks_semantic_fallback(
        self, mock_openai_cls, sample_schema: MetadataSchema
    ):
        """When SQL returns no rows and fallback=False, semantic_search follow-up is blocked."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: LLM chooses run_sql
        sql_tool_call = _make_tool_call(
            tool_call_id="call_sql",
            name="run_sql",
            arguments={"sql": "SELECT * FROM documents_metadata WHERE year > 1900;"},
        )
        first_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[sql_tool_call]
        )

        # Second call: LLM wants to fall back to semantic_search
        search_tool_call = _make_tool_call(
            tool_call_id="call_search",
            name="semantic_search",
            arguments={"query": "died after 1900"},
        )
        second_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[search_tool_call]
        )

        # Third call: LLM explains the data isn't available
        third_response = _make_openai_response(
            finish_reason="stop",
            content="This question cannot be answered precisely yet.",
        )

        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_executor = MagicMock()
        mock_executor.execute.return_value = "Query returned no rows."

        pipeline = QueryPipeline(
            llm_model="gpt-4o",
            tool_executor=mock_executor,
            schema=sample_schema,
            fallback=False,
        )

        answer = pipeline.query("Who died after 1900?")

        assert answer == "This question cannot be answered precisely yet."
        # Only the SQL call should have been executed, not the semantic_search
        mock_executor.execute.assert_called_once_with(
            "run_sql",
            {"sql": "SELECT * FROM documents_metadata WHERE year > 1900;"},
        )
        assert pipeline.last_fell_back is False

    @patch("meta_rag.query.pipeline.openai.OpenAI")
    def test_fallback_true_allows_fallback_and_sets_flag(
        self, mock_openai_cls, sample_schema: MetadataSchema
    ):
        """When SQL returns no rows and fallback=True, semantic_search follow-up proceeds."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: LLM chooses run_sql
        sql_tool_call = _make_tool_call(
            tool_call_id="call_sql",
            name="run_sql",
            arguments={"sql": "SELECT * FROM documents_metadata WHERE year > 1900;"},
        )
        first_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[sql_tool_call]
        )

        # Second call: LLM wants to fall back to semantic_search
        search_tool_call = _make_tool_call(
            tool_call_id="call_search",
            name="semantic_search",
            arguments={"query": "died after 1900"},
        )
        second_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[search_tool_call]
        )

        # Third call: LLM synthesizes from semantic results
        third_response = _make_openai_response(
            finish_reason="stop",
            content="Based on text search: Marie Curie, Alan Turing, Nikola Tesla.",
        )

        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = [
            "Query returned no rows.",  # SQL result
            "chunk1: Marie Curie died in 1934...",  # semantic_search result
        ]

        pipeline = QueryPipeline(
            llm_model="gpt-4o",
            tool_executor=mock_executor,
            schema=sample_schema,
            fallback=True,
        )

        answer = pipeline.query("Who died after 1900?")

        assert answer == "Based on text search: Marie Curie, Alan Turing, Nikola Tesla."
        assert mock_executor.execute.call_count == 2
        assert pipeline.last_fell_back is True

    @patch("meta_rag.query.pipeline.openai.OpenAI")
    def test_fallback_false_does_not_block_non_sql_followup(
        self, mock_openai_cls, sample_schema: MetadataSchema
    ):
        """When the first tool is semantic_search (not failed SQL), follow-up is allowed even with fallback=False."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: LLM chooses semantic_search
        search_tool_call = _make_tool_call(
            tool_call_id="call_search",
            name="semantic_search",
            arguments={"query": "Marie Curie"},
        )
        first_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[search_tool_call]
        )

        # Second call: LLM wants a follow-up SQL query
        sql_tool_call = _make_tool_call(
            tool_call_id="call_sql",
            name="run_sql",
            arguments={"sql": "SELECT year FROM documents_metadata WHERE author LIKE '%Curie%';"},
        )
        second_response = _make_openai_response(
            finish_reason="tool_calls", tool_calls=[sql_tool_call]
        )

        # Third call: LLM synthesizes
        third_response = _make_openai_response(
            finish_reason="stop",
            content="Marie Curie was born in 1867.",
        )

        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_executor = MagicMock()
        mock_executor.execute.side_effect = [
            "chunk1: Marie Curie biography...",  # semantic_search result
            json.dumps([{"year": 1867}]),  # SQL result
        ]

        pipeline = QueryPipeline(
            llm_model="gpt-4o",
            tool_executor=mock_executor,
            schema=sample_schema,
            fallback=False,
        )

        answer = pipeline.query("When was Marie Curie born?")

        assert answer == "Marie Curie was born in 1867."
        assert mock_executor.execute.call_count == 2
        assert pipeline.last_fell_back is False


# ---------------------------------------------------------------------------
# Integration test — Schema lifecycle
# ---------------------------------------------------------------------------


class TestSchemaLifecycle:
    @patch("meta_rag.ingestion.extractor.MetadataExtractor.extract")
    @patch("meta_rag.ingestion.extractor.MetadataExtractor.__init__", return_value=None)
    @patch("meta_rag.stores.vector.ChromaVectorStore.add")
    @patch("meta_rag.stores.vector.ChromaVectorStore.initialize")
    @patch("meta_rag.stores.vector.ChromaVectorStore.__init__", return_value=None)
    def test_schema_persists_and_reconstructs(
        self,
        mock_chroma_init,
        mock_chroma_initialize,
        mock_chroma_add,
        mock_extractor_init,
        mock_extractor_extract,
        tmp_path,
    ):
        """Create MetaRAG with explicit schema, ingest, then verify a second instance reconstructs the schema from DB."""
        from meta_rag import MetaRAG

        data_dir = str(tmp_path / "meta_rag_data")
        os.makedirs(data_dir, exist_ok=True)

        schema_fields = [
            MetadataField(name="author", type="text", description="Author name"),
            MetadataField(name="year", type="integer", description="Publication year"),
        ]

        # First instance: explicit schema
        rag1 = MetaRAG(
            llm_model="gpt-4o",
            schema=schema_fields,
            data_dir=data_dir,
        )

        assert rag1.schema is not None
        assert len(rag1.schema.fields) == 2
        assert rag1.schema.fields[0].name == "author"
        assert rag1.schema.fields[1].name == "year"

        # Simulate ingestion: initialize the relational store (creates the table)
        # and insert a row so the DB file exists with schema.
        rag1.relational_store.initialize(rag1.schema)
        rag1.relational_store.insert("doc1", {"author": "Alice", "year": 2023})

        # Second instance: no schema provided, should reconstruct from DB
        rag2 = MetaRAG(
            llm_model="gpt-4o",
            schema=None,
            data_dir=data_dir,
        )

        assert rag2.schema is not None
        assert len(rag2.schema.fields) == 2
        field_names = {f.name for f in rag2.schema.fields}
        assert field_names == {"author", "year"}
