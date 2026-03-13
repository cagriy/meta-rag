from __future__ import annotations

import json

from meta_rag.stores.base import RelationalStore, VectorStore


class ToolExecutor:
    """Executes tool calls against vector and relational stores."""

    def __init__(
        self, vector_store: VectorStore, relational_store: RelationalStore
    ) -> None:
        self.vector_store = vector_store
        self.relational_store = relational_store

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Dispatch a tool call and return a formatted string result.

        Args:
            tool_name: The name of the tool to invoke.
            arguments: A dictionary of keyword arguments for the tool.

        Returns:
            A human/LLM-readable string representation of the results.

        Raises:
            ValueError: If *tool_name* is not a recognised tool.
        """
        if tool_name == "semantic_search":
            return self._execute_semantic_search(arguments)
        if tool_name == "run_sql":
            return self._execute_sql(arguments)
        raise ValueError(f"Unknown tool: {tool_name}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _execute_semantic_search(self, arguments: dict) -> str:
        query: str = arguments["query"]
        top_k: int = arguments.get("top_k", 10)
        raw_filters: dict | None = arguments.get("filters")

        chroma_filters: dict | None = None
        if raw_filters:
            # Drop null values and SQL-style wildcards (%, _) that mean "any value"
            valid = {
                k: v
                for k, v in raw_filters.items()
                if v is not None and v != "%" and v != "_" and str(v).strip("%_") != ""
            }
            if len(valid) == 1:
                k, v = next(iter(valid.items()))
                chroma_filters = {k: {"$eq": v}}
            elif len(valid) > 1:
                chroma_filters = {"$and": [{k: {"$eq": v}} for k, v in valid.items()]}

        results = self.vector_store.search(
            query_text=query, top_k=top_k, filters=chroma_filters
        )

        if not results:
            return "No results found."

        parts: list[str] = []
        for i, result in enumerate(results, start=1):
            snippet = result.text[:500]
            if len(result.text) > 500:
                snippet += "..."
            parts.append(
                f"[{i}] chunk_id={result.chunk_id}  score={result.score:.4f}\n"
                f"    {snippet}"
            )
        return "\n\n".join(parts)

    def _execute_sql(self, arguments: dict) -> str:
        sql: str = arguments["sql"]
        rows = self.relational_store.execute_sql(sql)

        if not rows:
            return "Query returned no rows."

        # Render as a JSON array for straightforward LLM consumption.
        return json.dumps(rows, indent=2, default=str)
