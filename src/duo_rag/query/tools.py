from __future__ import annotations

from duo_rag.schema import MetadataSchema


class ToolBuilder:
    """Thin wrapper providing the query module its own entry point for tool generation."""

    @staticmethod
    def build_tools(schema: MetadataSchema) -> list[dict]:
        """Build OpenAI tool definitions by delegating to the schema."""
        return schema.to_tool_definitions()
