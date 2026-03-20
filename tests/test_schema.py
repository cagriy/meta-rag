"""Comprehensive tests for duo_rag.schema module."""

from __future__ import annotations

import pytest

from duo_rag.schema import MetadataField, MetadataSchema


# ---------------------------------------------------------------------------
# 1. MetadataField creation
# ---------------------------------------------------------------------------


class TestMetadataField:
    def test_creation_with_all_attributes(self):
        field = MetadataField(name="author", type="text", description="Author name")
        assert field.name == "author"
        assert field.type == "text"
        assert field.description == "Author name"

    def test_creation_integer_type(self):
        field = MetadataField(name="year", type="integer", description="Publication year")
        assert field.name == "year"
        assert field.type == "integer"
        assert field.description == "Publication year"

    def test_is_dataclass_instance(self):
        """MetadataField should behave as a dataclass (equality by value)."""
        a = MetadataField(name="x", type="text", description="d")
        b = MetadataField(name="x", type="text", description="d")
        assert a == b


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def text_and_integer_fields() -> list[MetadataField]:
    return [
        MetadataField(name="author", type="text", description="Author of the document"),
        MetadataField(name="year", type="integer", description="Publication year"),
    ]


@pytest.fixture()
def schema_with_fields(text_and_integer_fields: list[MetadataField]) -> MetadataSchema:
    return MetadataSchema(fields=text_and_integer_fields)


@pytest.fixture()
def empty_schema() -> MetadataSchema:
    return MetadataSchema(fields=[])


# ---------------------------------------------------------------------------
# 2. to_ddl() — text + integer fields
# ---------------------------------------------------------------------------


class TestToDDLWithFields:
    def test_starts_with_create_table_if_not_exists(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        assert ddl.startswith("CREATE TABLE IF NOT EXISTS documents_metadata (")

    def test_ends_with_closing_paren_and_semicolon(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        assert ddl.rstrip().endswith(");")

    def test_contains_fixed_columns(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        assert "id INTEGER PRIMARY KEY AUTOINCREMENT" in ddl
        assert "doc_id TEXT NOT NULL" in ddl
        assert "chunk_id" not in ddl

    def test_contains_dynamic_text_field(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        assert "author TEXT" in ddl

    def test_contains_dynamic_integer_field(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        assert "year INTEGER" in ddl

    def test_fixed_columns_appear_before_dynamic(self, schema_with_fields: MetadataSchema):
        ddl = schema_with_fields.to_ddl()
        id_pos = ddl.index("id INTEGER PRIMARY KEY AUTOINCREMENT")
        doc_id_pos = ddl.index("doc_id TEXT NOT NULL")
        author_pos = ddl.index("author TEXT")
        year_pos = ddl.index("year INTEGER")
        assert id_pos < doc_id_pos < author_pos < year_pos

    def test_full_ddl_snapshot(self, schema_with_fields: MetadataSchema):
        expected = (
            "CREATE TABLE IF NOT EXISTS documents_metadata (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    doc_id TEXT NOT NULL,\n"
            "    author TEXT,\n"
            "    year INTEGER\n"
            ");"
        )
        assert schema_with_fields.to_ddl() == expected


# ---------------------------------------------------------------------------
# 3. to_ddl() — empty schema (no custom fields)
# ---------------------------------------------------------------------------


class TestToDDLEmpty:
    def test_contains_only_fixed_columns(self, empty_schema: MetadataSchema):
        ddl = empty_schema.to_ddl()
        assert "id INTEGER PRIMARY KEY AUTOINCREMENT" in ddl
        assert "doc_id TEXT NOT NULL" in ddl
        assert "chunk_id" not in ddl

    def test_full_ddl_snapshot_empty(self, empty_schema: MetadataSchema):
        expected = (
            "CREATE TABLE IF NOT EXISTS documents_metadata (\n"
            "    id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
            "    doc_id TEXT NOT NULL\n"
            ");"
        )
        assert empty_schema.to_ddl() == expected

    def test_no_trailing_comma_after_last_fixed_column(self, empty_schema: MetadataSchema):
        ddl = empty_schema.to_ddl()
        # After "doc_id TEXT NOT NULL" there should be a newline then ");", no comma
        assert "doc_id TEXT NOT NULL\n);" in ddl


# ---------------------------------------------------------------------------
# 4. to_extraction_prompt()
# ---------------------------------------------------------------------------


class TestToExtractionPrompt:
    def test_includes_instruction_text(self, schema_with_fields: MetadataSchema):
        prompt = schema_with_fields.to_extraction_prompt()
        assert "Extract the following fields from this document chunk." in prompt
        assert "Return JSON with these fields (use null if not found):" in prompt

    def test_includes_field_names(self, schema_with_fields: MetadataSchema):
        prompt = schema_with_fields.to_extraction_prompt()
        assert "author" in prompt
        assert "year" in prompt

    def test_includes_field_types(self, schema_with_fields: MetadataSchema):
        prompt = schema_with_fields.to_extraction_prompt()
        assert "(text)" in prompt
        assert "(integer)" in prompt

    def test_includes_field_descriptions(self, schema_with_fields: MetadataSchema):
        prompt = schema_with_fields.to_extraction_prompt()
        assert "Author of the document" in prompt
        assert "Publication year" in prompt

    def test_field_line_format(self, schema_with_fields: MetadataSchema):
        prompt = schema_with_fields.to_extraction_prompt()
        assert "- author (text): Author of the document" in prompt
        assert "- year (integer): Publication year" in prompt

    def test_empty_schema_prompt(self, empty_schema: MetadataSchema):
        prompt = empty_schema.to_extraction_prompt()
        assert "Extract the following fields" in prompt
        # No field lines, just the header
        assert "- " not in prompt


# ---------------------------------------------------------------------------
# 5. to_tool_definitions()
# ---------------------------------------------------------------------------


class TestToToolDefinitions:
    def test_returns_exactly_two_tools(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        assert len(tools) == 2

    # -- semantic_search tool --

    def test_first_tool_is_semantic_search(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        first = tools[0]
        assert first["type"] == "function"
        assert first["function"]["name"] == "semantic_search"

    def test_semantic_search_query_is_required(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        params = tools[0]["function"]["parameters"]
        assert params["required"] == ["query"]

    def test_semantic_search_has_query_property(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        props = tools[0]["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"

    def test_semantic_search_has_filters_with_field_properties(
        self, schema_with_fields: MetadataSchema
    ):
        tools = schema_with_fields.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["type"] == "object"
        assert "author" in filters["properties"]
        assert "year" in filters["properties"]

    def test_semantic_search_has_top_k(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        props = tools[0]["function"]["parameters"]["properties"]
        assert "top_k" in props
        assert props["top_k"]["type"] == "integer"

    def test_semantic_search_filter_descriptions(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["properties"]["author"]["description"] == "Author of the document"
        assert filters["properties"]["year"]["description"] == "Publication year"

    # -- run_sql tool --

    def test_second_tool_is_run_sql(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        second = tools[1]
        assert second["type"] == "function"
        assert second["function"]["name"] == "run_sql"

    def test_run_sql_sql_is_required(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        params = tools[1]["function"]["parameters"]
        assert params["required"] == ["sql"]

    def test_run_sql_has_sql_property(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        props = tools[1]["function"]["parameters"]["properties"]
        assert "sql" in props
        assert props["sql"]["type"] == "string"

    def test_run_sql_description_contains_table_schema(
        self, schema_with_fields: MetadataSchema
    ):
        tools = schema_with_fields.to_tool_definitions()
        desc = tools[1]["function"]["description"]
        assert "Table: documents_metadata(" in desc
        assert "id INTEGER PRIMARY KEY" in desc
        assert "doc_id TEXT" in desc
        assert "chunk_id" not in desc
        assert "author TEXT" in desc
        assert "year INTEGER" in desc

    # -- empty schema tool definitions --

    def test_empty_schema_returns_two_tools(self, empty_schema: MetadataSchema):
        tools = empty_schema.to_tool_definitions()
        assert len(tools) == 2

    def test_empty_schema_semantic_search_filters_empty(self, empty_schema: MetadataSchema):
        tools = empty_schema.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["properties"] == {}

    def test_empty_schema_run_sql_has_fixed_columns_only(self, empty_schema: MetadataSchema):
        tools = empty_schema.to_tool_definitions()
        desc = tools[1]["function"]["description"]
        assert "id INTEGER PRIMARY KEY" in desc
        assert "doc_id TEXT" in desc
        assert "chunk_id" not in desc


# ---------------------------------------------------------------------------
# 6. Filter type mapping
# ---------------------------------------------------------------------------


class TestFilterTypeMapping:
    def test_text_field_maps_to_string(self):
        schema = MetadataSchema(
            fields=[MetadataField(name="title", type="text", description="Title")]
        )
        tools = schema.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["properties"]["title"]["type"] == "string"

    def test_integer_field_maps_to_integer(self):
        schema = MetadataSchema(
            fields=[MetadataField(name="page_count", type="integer", description="Pages")]
        )
        tools = schema.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["properties"]["page_count"]["type"] == "integer"

    def test_mixed_types_mapped_correctly(self, schema_with_fields: MetadataSchema):
        tools = schema_with_fields.to_tool_definitions()
        filters = tools[0]["function"]["parameters"]["properties"]["filters"]
        assert filters["properties"]["author"]["type"] == "string"
        assert filters["properties"]["year"]["type"] == "integer"
