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

    def query(self, question: str) -> str:
        system_prompt = (
            "You are a helpful assistant that answers questions about a document collection. "
            "Use the available tools to find information. "
            "For quantitative questions (counts, averages, comparisons), prefer run_sql. "
            "For qualitative questions (descriptions, explanations), prefer semantic_search."
        )

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        # First LLM call: decide whether to use tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
        )

        choice = response.choices[0]

        # If the LLM answered directly without tools, return as-is
        if choice.finish_reason == "stop":
            return choice.message.content

        # Execute each tool call and collect results
        tool_results: list[dict] = []
        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            result = self.tool_executor.execute(name, arguments)
            tool_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        # Build messages for the second LLM call
        messages.append(choice.message)
        for tool_result in tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result["content"],
                }
            )

        # Second LLM call: synthesize a natural language answer from tool results
        synthesis_response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return synthesis_response.choices[0].message.content
