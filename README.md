# meta-rag

Extends RAG with structured query capabilities using dual vector + relational storage.

## The Problem

Structured queries over a document corpus require two things that traditional RAG does not provide:

**1. A schema defined upfront.** To answer aggregation questions ("How many people are from England?", "What is the average birth year?"), you need structured metadata in a relational store. That means deciding in advance which fields matter — before you fully understand your data. Get it wrong and you either re-ingest everything or can't answer the query at all.

**2. Ongoing schema learning.** As users ask new kinds of questions, the right schema evolves. Traditional RAG solutions have no mechanism to detect that a query requires a field that doesn't exist yet, add it, and retroactively populate it from already-ingested documents. Every gap requires a manual schema change and a full re-ingest.

meta-rag addresses both: it can auto-discover an initial schema from your documents, and it continuously monitors queries for schema gaps — adding fields and backfilling them from stored chunks without re-ingesting source files.

## How It Works

meta-rag maintains two parallel stores for every ingested document:


| Store            | Backend  | Used for                         |
| ------------------ | ---------- | ---------------------------------- |
| Vector store     | ChromaDB | Semantic similarity search       |
| Relational store | SQLite   | Aggregation, filtering, counting |

At query time, an LLM uses **tool-calling** to decide which backend to hit — or both — based on the question. The LLM calls `semantic_search` (ChromaDB) or `run_sql` (SQLite) with the full table schema embedded in the tool description, so it can write correct SQL automatically.

### Schema

You define a list of `MetadataField` objects describing the structured data to extract from each document. The LLM extracts those fields during ingestion and stores them in SQLite. If you skip the schema, meta-rag auto-discovers one by sampling a few documents.

## Key Features

- **Dual-store routing** — LLM picks the right backend per question
- **Auto schema discovery** — infer fields from document samples when no schema is provided
- **Schema evolution** — detect missing fields mid-session (`evolve=True`) and add them live
- **Incremental ingestion** — hash-based deduplication skips unchanged documents
- **Backfill** — populate newly added fields from already-stored chunks without re-ingesting

## Installation

```bash
uv add meta-rag
# or
pip install meta-rag
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
# or add it to a .env file in your project root
```

## Quick Start

```python
from meta_rag import MetaRAG, MetadataField

# 1. Define a schema (or omit for auto-discovery)
schema = [
    MetadataField(name="birthplace",    type="text",    description="Person's place of birth"),
    MetadataField(name="occupation",    type="text",    description="Primary occupation or field"),
    MetadataField(name="year_of_birth", type="integer", description="Year the person was born"),
]

# 2. Initialize
rag = MetaRAG(
    llm_model="gpt-4o",
    schema=schema,
    data_dir="./my_data",
)

# 3. Ingest documents (incremental — safe to call repeatedly)
stats = rag.ingest("./documents/")
print(stats)  # {"new": 10, "changed": 0, "unchanged": 0}

# 4. Query — routing is automatic
print(rag.query("What did Marie Curie discover?"))          # → semantic search
print(rag.query("How many people were born after 1800?"))   # → SQL
print(rag.query("What is the most common occupation?"))     # → SQL aggregation
```

### Auto schema discovery

```python
# No schema provided — meta-rag infers fields from a document sample on first ingest
rag = MetaRAG(llm_model="gpt-4o", data_dir="./my_data")
rag.ingest("./documents/")
print([f.name for f in rag.schema.fields])  # e.g. ["name", "birthplace", "occupation", ...]
```

### Schema evolution

```python
# Pass evolve=True to detect and add missing fields on the fly
answer = rag.query("How many people died before 1900?", evolve=True)
# answer may include:
# [Schema Gap Detected] 'year_of_death' has been added. Run backfill() to populate it.

# Populate the new field from all stored chunks
result = rag.backfill()
print(result)  # {"populated": ["year_of_death"], "pruned": []}
```

### Add a field manually

```python
rag.add_field(MetadataField(
    name="nationality",
    type="text",
    description="Person's nationality",
))
rag.backfill()
```

## Running the Example

The repository includes an example script that ingests a set of biographical `.txt` files and opens an interactive query loop.

```bash
# Ingest documents and enter interactive mode
uv run python examples/example_usage.py

# Run pre-defined demo queries first, then enter interactive mode
uv run python examples/example_usage.py --test

# Also print the generated SQL alongside each answer
uv run python examples/example_usage.py --test --verbose
```

**Interactive mode commands:**


| Input                 | Action                                                        |
| ----------------------- | --------------------------------------------------------------- |
| Any question          | Query with`evolve=True` — schema gaps detected automatically |
| `/backfill`           | Populate newly added fields from stored chunks                |
| `/ingest`             | Re-ingest documents from`examples/documents/`                 |
| `quit` / `exit` / `q` | Exit                                                          |

## API Reference

### `MetaRAG`

```python
MetaRAG(
    llm_model: str = "gpt-4o",
    schema: list[MetadataField] | None = None,
    data_dir: str = "./meta_rag_data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    vector_store: ChromaVectorStore | None = None,
    relational_store: SQLiteRelationalStore | None = None,
)
```


| Parameter          | Default             | Description                                                              |
| -------------------- | --------------------- | -------------------------------------------------------------------------- |
| `llm_model`        | `"gpt-4o"`          | OpenAI model for extraction and querying                                 |
| `schema`           | `None`              | List of`MetadataField`; auto-discovered on first ingest if omitted       |
| `data_dir`         | `"./meta_rag_data"` | Directory for ChromaDB and SQLite persistence                            |
| `chunk_size`       | `1000`              | Max characters per text chunk                                            |
| `chunk_overlap`    | `200`               | Character overlap between consecutive chunks                             |
| `vector_store`     | `None`              | Custom`ChromaVectorStore` (default created in `data_dir` if omitted)     |
| `relational_store` | `None`              | Custom`SQLiteRelationalStore` (default created in `data_dir` if omitted) |

### `ingest(path, on_progress=None) → dict`

Ingest a file path or list of paths. Skips unchanged documents (hash-based). Auto-discovers schema if none is set.

- `path` — `str` or `list[str]`
- `on_progress` — optional `(current: int, total: int) -> None` callback

Returns `{"new": int, "changed": int, "unchanged": int}`.

### `query(question, evolve=False, history=None) → str`

Ask a question. The LLM routes to semantic search, SQL, or both.

- `evolve=True` — after answering, check for schema gaps and add detected fields automatically
- `history` — list of prior `{"role": ..., "content": ...}` messages for multi-turn conversation

Use `rag.last_history` to get the updated history after each call for follow-up questions. Use `rag.last_sql` to inspect the SQL that was generated (if any).

### `backfill(on_progress=None) → dict`

Extract values for all unpopulated fields from already-stored chunks. Prunes fields that remain entirely NULL after backfill.

- `on_progress` — optional `(current: int, total: int) -> None` callback

Returns `{"populated": [field names], "pruned": [field names]}`.

### `add_field(field: MetadataField) → None`

Add a new field to the live schema and the SQLite database. Call `backfill()` afterward to populate it from existing documents.

## Project Structure

```
src/meta_rag/
├── __init__.py          # MetaRAG facade — public API
├── schema.py            # MetadataField, MetadataSchema, SchemaEvolutionResult
├── ingestion/
│   ├── chunker.py       # Text splitting
│   ├── extractor.py     # LLM metadata extraction
│   └── pipeline.py      # Ingestion orchestration + schema discovery
├── query/
│   ├── pipeline.py      # Query orchestration + tool-call loop
│   ├── executor.py      # Tool execution (semantic_search, run_sql)
│   └── tools.py         # Tool definitions
└── stores/
    ├── base.py          # SearchResult dataclass
    ├── vector.py        # ChromaDB vector store
    └── relational.py    # SQLite relational store
```
