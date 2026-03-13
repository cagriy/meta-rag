from __future__ import annotations

import json

import openai

from meta_rag.query.executor import ToolExecutor
from meta_rag.schema import MetadataSchema


class QueryPipeline:
    def __init__(
        self,
        llm_model: str,
        tool_executor: ToolExecutor,
        schema: MetadataSchema,
    ) -> None:
        self.client = openai.OpenAI()
        self.model = llm_model
        self.tool_executor = tool_executor
        self.schema = schema
        self.tools = schema.to_tool_definitions()
        self.last_sql: str | None = None
        self.last_sql_returned_results: bool = False
        self.messages: list[dict] = []

    def query(self, question: str, history: list[dict] | None = None) -> str:
        available_columns = ", ".join(f.name for f in self.schema.fields)
        system_prompt = (
            "You are a helpful assistant that answers questions about a document collection. "
            "Use the available tools to find information. "
            "For quantitative questions (counts, averages, comparisons), prefer run_sql. "
            "For qualitative questions (descriptions, explanations), prefer semantic_search. "
            "When writing SQL for text fields, use LIKE with wildcards for partial matching "
            "(e.g. WHERE birthplace LIKE '%England%') rather than exact equality, "
            "since stored values are full location or description strings.\n\n"
            f"The available metadata columns are: {available_columns}. "
            "IMPORTANT: Only reference these exact columns in SQL. "
            "If the information needed to answer the question is not available as a column, "
            "do NOT approximate with an unrelated column — use semantic_search instead."
        )

        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        # First LLM call: decide whether to use tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
        )

        choice = response.choices[0]

        # If the LLM answered directly without tools, return as-is
        if choice.finish_reason == "stop":
            messages.append({"role": "assistant", "content": choice.message.content})
            self.messages = messages[1:]  # strip system message for history
            return choice.message.content

        # Execute each tool call and collect results
        self.last_sql = None
        self.last_sql_returned_results = False
        tool_results: list[dict] = []
        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            if name == "run_sql":
                self.last_sql = arguments.get("sql")
            result = self.tool_executor.execute(name, arguments)
            if name == "run_sql" and result != "Query returned no rows.":
                self.last_sql_returned_results = True
            tool_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Build messages with tool results
        messages.append(choice.message)
        for tool_result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result["content"],
                }
            )

        # Second LLM call: synthesize or make a follow-up tool call
        synthesis_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
        )

        synthesis_choice = synthesis_response.choices[0]

        # If the LLM wants to make one more tool call (e.g. SQL returned empty,
        # falling back to semantic_search), execute it and do a final synthesis
        if synthesis_choice.finish_reason == "tool_calls":
            followup_results: list[dict] = []
            for tool_call in synthesis_choice.message.tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                if name == "run_sql":
                    self.last_sql = arguments.get("sql")
                result = self.tool_executor.execute(name, arguments)
                if name == "run_sql" and result != "Query returned no rows.":
                    self.last_sql_returned_results = True
                followup_results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )
            messages.append(synthesis_choice.message)
            for tool_result in followup_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_result["tool_call_id"],
                        "content": tool_result["content"],
                    }
                )
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            final_content = final_response.choices[0].message.content
            messages.append({"role": "assistant", "content": final_content})
            self.messages = messages[1:]  # strip system message for history
            return final_content

        synthesis_content = synthesis_choice.message.content
        messages.append({"role": "assistant", "content": synthesis_content})
        self.messages = messages[1:]  # strip system message for history
        return synthesis_content
