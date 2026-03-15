from __future__ import annotations

import json

import openai

from meta_rag.prompts import DEFAULT_METADATA_EXTRACTION_PROMPT, DEFAULT_SCHEMA_DISCOVERY_PROMPT
from meta_rag.schema import MetadataField, MetadataSchema


class MetadataExtractor:
    """Extracts structured metadata from text using an LLM."""

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        extraction_prompt: str = DEFAULT_METADATA_EXTRACTION_PROMPT,
        discovery_prompt: str = DEFAULT_SCHEMA_DISCOVERY_PROMPT,
    ) -> None:
        self.client = openai.OpenAI()
        self.llm_model = llm_model
        self.extraction_prompt = extraction_prompt
        self.discovery_prompt = discovery_prompt

    def extract(self, text: str, schema: MetadataSchema) -> dict:
        """Extract metadata fields from a text chunk according to the given schema.

        Args:
            text: The document text to extract metadata from.
            schema: The metadata schema defining which fields to extract.

        Returns:
            A dictionary mapping field names to their extracted values.
        """
        system_message = self.extraction_prompt.format(
            extraction_fields=schema.to_extraction_prompt(),
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)

        result: dict = {}
        for field in schema.fields:
            value = raw.get(field.name)
            if value is None:
                result[field.name] = None
            elif field.type == "integer":
                try:
                    result[field.name] = int(value)
                except (ValueError, TypeError):
                    result[field.name] = None
            elif isinstance(value, list):
                result[field.name] = ", ".join(str(v) for v in value if v is not None) or None
            else:
                result[field.name] = str(value)

        return result

    def discover_schema(self, sample_texts: list[str]) -> MetadataSchema:
        """Analyze sample documents and propose a metadata schema.

        Args:
            sample_texts: A list of representative document texts to analyze.

        Returns:
            A MetadataSchema describing the discovered fields.
        """
        system_message = self.discovery_prompt

        numbered_samples = "\n\n".join(
            f"--- Sample {i + 1} ---\n{text}" for i, text in enumerate(sample_texts)
        )

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": numbered_samples},
            ],
            response_format={"type": "json_object"},
        )

        raw = json.loads(response.choices[0].message.content)

        fields = [
            MetadataField(
                name=f["name"],
                type=f["type"],
                description=f["description"],
            )
            for f in raw["fields"]
        ]

        return MetadataSchema(fields=fields)
