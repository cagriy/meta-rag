from __future__ import annotations

import json

import openai

from duo_rag.prompts import DEFAULT_QUERY_SYSTEM_PROMPT
from duo_rag.query.executor import ToolExecutor
from duo_rag.schema import MetadataSchema


class QueryPipeline:
    def __init__(
        self,
        llm_model: str,
        tool_executor: ToolExecutor,
        schema: MetadataSchema,
        query_system_prompt: str = DEFAULT_QUERY_SYSTEM_PROMPT,
        fallback: bool = False,
    ) -> None:
        self.client = openai.OpenAI()
        self.model = llm_model
        self.tool_executor = tool_executor
        self.schema = schema
        self.tools = schema.to_tool_definitions()
        self.query_system_prompt = query_system_prompt
        self.fallback = fallback
        self.last_sql: str | None = None
        self.last_sql_returned_results: bool = False
        self.last_fell_back: bool = False
        self.messages: list[dict] = []

    def query(self, question: str, history: list[dict] | None = None) -> str:
        available_columns = ", ".join(f.name for f in self.schema.fields)
        system_prompt = self.query_system_prompt.format(
            available_columns=available_columns,
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
            parallel_tool_calls=False,
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
        self.last_fell_back = False
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
        # falling back to semantic_search), decide whether to allow it
        if synthesis_choice.finish_reason == "tool_calls":
            sql_failed = self.last_sql is not None and not self.last_sql_returned_results

            if sql_failed and not self.fallback:
                # Block the fallback — ask LLM to explain without tools
                messages.append(
                    {
                        "role": "user",
                        "content": "The SQL query returned no results.",
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

            # Allow the follow-up tool call(s)
            if sql_failed:
                self.last_fell_back = True

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
