# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-03-15

### Changed
- All user-facing response messages are now LLM-generated via existing prompts instead of hardcoded strings
- Schema gap detection prompt returns `message` and `unavailable_message` fields in its JSON response
- Query system prompt includes behavioral instructions for SQL failure and semantic search fallback scenarios
- User-facing messages use friendly language and suggest checking back later instead of exposing technical details like `backfill()` or field names

### Added
- `unavailable_message` field on `SchemaEvolutionResult` for self-contained user-facing explanations

## [0.1.4] - 2026-03-15

### Added
- Full document retrieval for semantic search — when a chunk matches, all chunks from the same document are fetched, ordered by `chunk_index`, and the complete document text is returned to the LLM
- `chunk_index` and `total_chunks` stored as metadata on every chunk for correct document reconstruction
- `get_document_chunks()` method on vector store to retrieve all chunks for a given document
- Three new multi-chunk example documents (Faraday, Hypatia, Ramanujan) demonstrating cross-chunk retrieval

### Changed
- Semantic search results now return full reconstructed documents instead of 500-character truncated snippets
- Schema gap detection prompt tightened to avoid flagging content questions (e.g. "Who discovered X?") as schema gaps when they are well-served by semantic search

## [0.1.3] - 2026-03-13

### Added
- `extraction_model` parameter on `MetaRAG` (defaults to `gpt-4o-mini`) — separates the model used for metadata extraction and schema discovery from the model used for query routing and schema gap detection
- `--no-schema` and `--reset` flags added to the example script

### Changed
- Both `llm_model` and `extraction_model` now default to `gpt-4o-mini`
- Metadata extraction now runs once per document instead of once per chunk, reducing LLM calls proportional to chunk count
- `discover_schema` reads full document text instead of chunked samples; uses sqrt-scaled sampling (5–50 docs) instead of a fixed count
- Available schema columns are injected into the query system prompt; LLM is instructed to fall back to `semantic_search` rather than approximate with an unrelated column when the required field is missing from the schema

## [0.1.2] - 2026-03-13

### Added
- `.env.example` documenting required environment variables

### Changed
- Author email in `pyproject.toml` updated to GitHub noreply address

## [0.1.1] - 2026-02-28

### Fixed
- `ChromaVectorStore.add()` now filters out `None` metadata values before passing to ChromaDB (which only accepts `str`, `int`, `float`, `bool`)

### Changed
- Example script wipes `example_data/` before ingesting to ensure a clean run
- Example script enters an interactive query loop after preset questions finish

## [0.1.0] - 2026-02-28

### Added
- `MetaRAG` facade with schema lifecycle management (auto-discover or manual schema)
- `MetadataSchema` and `MetadataField` for defining structured extraction fields
- `ChromaVectorStore` — embedded ChromaDB implementation for semantic search
- `SQLiteRelationalStore` — embedded SQLite implementation with SQL safety (SELECT-only, read-only connection)
- `IngestionPipeline` — chunk → LLM extract → dual-write to both stores
- `QueryPipeline` — two-step LLM tool-calling flow routing queries to the right backend
- `Chunker` — character-based text chunking with configurable size and overlap
- `MetadataExtractor` — LLM-based metadata extraction and schema auto-discovery
- `ToolBuilder` and `ToolExecutor` for OpenAI function-calling integration
- Schema persistence: schema is read back from SQLite on subsequent sessions
- 62 unit and integration tests
- Example usage script with 8 sample biography documents
