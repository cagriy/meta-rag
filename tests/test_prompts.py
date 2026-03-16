from __future__ import annotations

from unittest.mock import MagicMock, patch

from meta_rag import MetaRAG, PromptConfig
from meta_rag.prompts import (
    DEFAULT_METADATA_EXTRACTION_PROMPT,
    DEFAULT_QUERY_SYSTEM_PROMPT,
    DEFAULT_SCHEMA_DISCOVERY_PROMPT,
    DEFAULT_SCHEMA_GAP_DETECTION_PROMPT,
    PromptConfig as PromptConfigDirect,
)
from meta_rag.schema import MetadataField, MetadataSchema


class TestPromptConfigDefaults:
    def test_defaults_match_module_constants(self):
        config = PromptConfig()
        assert config.query_system_prompt == DEFAULT_QUERY_SYSTEM_PROMPT
        assert config.metadata_extraction_prompt == DEFAULT_METADATA_EXTRACTION_PROMPT
        assert config.schema_discovery_prompt == DEFAULT_SCHEMA_DISCOVERY_PROMPT
        assert config.schema_gap_detection_prompt == DEFAULT_SCHEMA_GAP_DETECTION_PROMPT

    def test_query_system_prompt_format(self):
        config = PromptConfig()
        result = config.query_system_prompt.format(available_columns="name, age")
        assert "name, age" in result

    def test_metadata_extraction_prompt_format(self):
        config = PromptConfig()
        result = config.metadata_extraction_prompt.format(extraction_fields="field1, field2")
        assert "field1, field2" in result

    def test_schema_discovery_prompt_no_placeholders(self):
        config = PromptConfig()
        # Should not raise — no format placeholders
        result = config.schema_discovery_prompt.format()
        assert "schema discovery" in result

    def test_schema_gap_detection_prompt_format_with_escaped_braces(self):
        config = PromptConfig()
        result = config.schema_gap_detection_prompt.format(
            fields_text="year (integer): Year of birth",
            unpopulated_fields_text="(none)",
        )
        # Escaped braces should render as literal braces
        assert "{name (snake_case)" in result
        assert "year (integer): Year of birth" in result

    def test_schema_gap_detection_prompt_includes_message_fields(self):
        config = PromptConfig()
        prompt = config.schema_gap_detection_prompt
        assert "- message (string)" in prompt
        assert "- unavailable_message (string)" in prompt

    def test_query_system_prompt_includes_fallback_instructions(self):
        config = PromptConfig()
        prompt = config.query_system_prompt
        assert "SQL query" in prompt
        assert "check back later" in prompt
        assert "semantic search" in prompt

    def test_custom_override(self):
        config = PromptConfig(query_system_prompt="Custom: {available_columns}")
        assert config.query_system_prompt == "Custom: {available_columns}"
        assert config.query_system_prompt.format(available_columns="x") == "Custom: x"
        # Other fields remain default
        assert config.metadata_extraction_prompt == DEFAULT_METADATA_EXTRACTION_PROMPT


class TestPromptConfigThreading:
    @patch("meta_rag.ingestion.extractor.openai.OpenAI")
    def test_custom_prompts_thread_to_extractor(self, mock_openai_cls):
        custom_extraction = "Custom extraction: {extraction_fields}"
        custom_discovery = "Custom discovery prompt"
        config = PromptConfig(
            metadata_extraction_prompt=custom_extraction,
            schema_discovery_prompt=custom_discovery,
        )
        rag = MetaRAG(prompts=config, data_dir="/tmp/test_prompt_threading")

        from meta_rag.ingestion.extractor import MetadataExtractor

        extractor = MetadataExtractor(
            llm_model="test",
            extraction_prompt=rag.prompts.metadata_extraction_prompt,
            discovery_prompt=rag.prompts.schema_discovery_prompt,
        )
        assert extractor.extraction_prompt == custom_extraction
        assert extractor.discovery_prompt == custom_discovery

    def test_custom_prompts_thread_to_query_pipeline(self):
        custom_query = "Custom query: {available_columns}"
        config = PromptConfig(query_system_prompt=custom_query)
        rag = MetaRAG(prompts=config, data_dir="/tmp/test_prompt_threading2")
        assert rag.prompts.query_system_prompt == custom_query

    def test_default_prompts_when_none(self):
        rag = MetaRAG(data_dir="/tmp/test_prompt_default")
        assert rag.prompts.query_system_prompt == DEFAULT_QUERY_SYSTEM_PROMPT
        assert rag.prompts.metadata_extraction_prompt == DEFAULT_METADATA_EXTRACTION_PROMPT
        assert rag.prompts.schema_discovery_prompt == DEFAULT_SCHEMA_DISCOVERY_PROMPT
        assert rag.prompts.schema_gap_detection_prompt == DEFAULT_SCHEMA_GAP_DETECTION_PROMPT
