# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
