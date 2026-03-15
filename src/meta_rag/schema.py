from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetadataField:
    """A single metadata field definition."""

    name: str
    type: str  # "text" or "integer"
    description: str


@dataclass
class SchemaEvolutionResult:
    """Result of a schema gap detection check."""

    gap_detected: bool
    reasoning: str
    proposed_field: MetadataField | None
    message: str  # user-facing message
    unavailable_message: str  # user-facing message when data isn't available yet


TYPE_MAP = {
    "text": "TEXT",
    "integer": "INTEGER",
}


class MetadataSchema:
    """Schema definition for structured metadata extracted from documents."""

    TABLE_NAME = "documents_metadata"

    def __init__(self, fields: list[MetadataField]) -> None:
        self.fields = fields

    def add_field(self, field: MetadataField) -> None:
        """Append a new field to the schema."""
        self.fields.append(field)

    def remove_field(self, field_name: str) -> None:
        """Remove a field from the schema by name."""
        self.fields = [f for f in self.fields if f.name != field_name]

    def to_ddl(self) -> str:
        """Generate SQL CREATE TABLE statement for the metadata schema."""
        lines = [
            "id INTEGER PRIMARY KEY AUTOINCREMENT",
            "doc_id TEXT NOT NULL",
        ]
        for field in self.fields:
            sql_type = TYPE_MAP[field.type]
            lines.append(f"{field.name} {sql_type}")

        columns = ",\n    ".join(lines)
        return (
            f"CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (\n"
            f"    {columns}\n"
            f");"
        )

    def to_extraction_prompt(self) -> str:
        """Generate a prompt instructing an LLM to extract the schema fields."""
        field_lines = []
        for field in self.fields:
            field_lines.append(f"- {field.name} ({field.type}): {field.description}")

        fields_text = "\n".join(field_lines)
        return (
            "Extract the following fields from this document chunk.\n"
            "Return JSON with these fields (use null if not found):\n"
            "\n"
            f"{fields_text}"
        )

    def to_tool_definitions(self) -> list[dict]:
        """Generate OpenAI function-calling tool definitions for semantic search and SQL."""
        # Build filter properties for the semantic_search tool
        filter_properties: dict = {}
        for field in self.fields:
            if field.type == "text":
                json_type = "string"
            else:
                json_type = "integer"
            filter_properties[field.name] = {
                "type": json_type,
                "description": field.description,
            }

        semantic_search_tool = {
            "type": "function",
            "function": {
                "name": "semantic_search",
                "description": "Search documents using semantic similarity with optional metadata filters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for semantic similarity matching.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional metadata filters to narrow results.",
                            "properties": filter_properties,
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return.",
                        },
                    },
                    "required": ["query"],
                },
            },
        }

        # Build the table schema string for the run_sql tool description
        schema_parts = [
            "id INTEGER PRIMARY KEY",
            "doc_id TEXT",
        ]
        for field in self.fields:
            sql_type = TYPE_MAP[field.type]
            schema_parts.append(f"{field.name} {sql_type}")

        schema_str = ", ".join(schema_parts)
        table_schema = f"Table: {self.TABLE_NAME}({schema_str})"

        run_sql_tool = {
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": (
                    f"Run a SQL query against the metadata database. {table_schema}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "The SQL query to execute.",
                        },
                    },
                    "required": ["sql"],
                },
            },
        }

        return [semantic_search_tool, run_sql_tool]
