# MetaRAG

Imagine a corpus of documents with scientist biographies.

The traditional RAG works fine until you ask questions like:

"Who was born before 1800?"

"How many are mathematicians?"

"List names and birthdays for mathematicians"

These result in an incomplete answer due to top-k with no signs of incompleteness.

For an initial corpus it is possible to improve this problem by extracting metadata for a predetermined set of fields. This approach has two problems:

1. One has to predict all the questions that can be asked against the corpus upfront.
2. Constantly revising that prediction as the documents change, e.g. adding nobel prizes later, or extending the document set to contain artists.

MetaRAG solves both problems by:

1. An initial metadata (schema) discovery before the first ingestion
2. Self-update schema with candidate fields when it fails to answer a question

A periodic "backfill" run then extracts and populates the candidate fields, or prunes them if the information is not contained within the corpus. If the backfill is running nightly, a question that has failed today, gets answered correctly tomorrow.

## How It Works

MetaRAG maintains two parallel stores for every ingested document:


| Store            | Backend  | Used for                         |
| ------------------ | ---------- | ---------------------------------- |
| Vector store     | ChromaDB | Semantic similarity search       |
| Relational store | SQLite   | Aggregation, filtering, counting |

At query time, an LLM uses **tool-calling** to decide which backend to hit — or both — based on the question. The LLM calls `semantic_search` (ChromaDB) or `run_sql` (SQLite) with the full table schema embedded in the tool description, so it can write correct SQL automatically.

### Schema

MetaRAG allows you define an initial schema, if you prefer to cover the most predictable fields and let the rest evolve based on user queries. If you skip the schema, MetaRAG auto-discovers one by sampling a configurable subset of documents.

## Key Features

- **Dual-store routing** — LLM picks the right backend per question
- **Auto schema discovery** — infer fields from document samples when no schema is provided
- **Schema evolution** — detect missing fields mid-session (`evolve=True`) and add them live
- **Controlled fallback** — blocks incomplete semantic search answers for aggregate questions; SQL failures prompt schema evolution instead of silent top-k fallback
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
    llm_model="gpt-5-mini",
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
# No schema provided — MetaRAG infers fields from a document sample on first ingest
rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_data")
rag.ingest("./documents/")
print([f.name for f in rag.schema.fields])  # e.g. ["name", "birthplace", "occupation", ...]
```

### Schema evolution

```python
# By default, if SQL can't answer a question (missing column), MetaRAG won't
# fall back to semantic search — avoiding misleading partial answers.
answer = rag.query("How many people died before 1900?", evolve=True)
# → explains the data isn't available yet as structured metadata
# → [Schema Gap Detected] 'year_of_death' has been added. Run backfill() to populate it.

# Populate the new field from all stored chunks
result = rag.backfill()
print(result)  # {"populated": ["year_of_death"], "pruned": []}

# Now the same query works precisely via SQL
answer = rag.query("How many people died before 1900?")

# If you prefer partial answers over no answer, enable fallback:
answer = rag.query("How many people died before 1900?", fallback=True)
# → returns top-k semantic results with a warning about incompleteness
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

## Evaluation

The repository includes an evaluation suite that tests MetaRAG's core capabilities — semantic search, SQL generation, schema evolution, backfill, and conversational follow-ups — using a combination of LLM-judge scoring and deterministic checks.

### How it works

Tests are defined in `examples/eval_tests.yaml` as a sequence of **stages**, each containing one or more test cases. Every test specifies a question, expected behavior (e.g. SQL usage, expected keywords), and judge criteria. The eval runner:

1. Ingests the sample documents from `examples/documents/`
2. Executes each test stage in order (basic queries → schema evolution → backfill → conversation)
3. Scores each answer on two axes:
   - **LLM judge** (0.0–1.0) — evaluates correctness, completeness, and relevance
   - **Deterministic checks** — validates SQL usage, expected/excluded keywords, schema gap detection, etc.
4. A test **passes** if the judge score is ≥ 0.7 **and** all deterministic checks pass

### Running the eval

```bash
# Basic run (reuses existing eval_data if present)
uv run --group eval python examples/run_eval.py

# Clean run — delete eval_data and start fresh
uv run --group eval python examples/run_eval.py --reset

# Verbose — print answers and judge reasoning
uv run --group eval python examples/run_eval.py --verbose

# Save a detailed JSON report
uv run --group eval python examples/run_eval.py --save-report eval_report.json
```

### Test stages

| Stage | What it tests | Example question |
| --- | --- | --- |
| Basic queries | Semantic search and SQL routing | "Who was born after 1800?" |
| Schema evolution | Gap detection and field auto-addition | "Who has died after 1900?" |
| Backfill | Populating newly added fields | Re-asks post-backfill query |
| Conversation | Multi-turn context preservation | "Total mathematicians?" → "Who are they?" |

### Adding or modifying tests

Edit `examples/eval_tests.yaml`. Each test case supports:

```yaml
- id: my_test
  question: "How many scientists were born in England?"
  type: quantitative          # factual | quantitative | schema_evolution | conversational
  evolve: false               # trigger schema evolution?
  judge_criteria: "Should return the correct count"
  expect_sql: true            # assert SQL was used
  expected_keywords: ["England"]
  expected_names: ["Newton", "Darwin", "Faraday"]
  excluded_names: []
  expect_gap_detected: false
  save_history: false         # save conversation history for follow-up tests
  continues_from: ""          # test id to continue conversation from
```

## API Reference

### `MetaRAG`

```python
MetaRAG(
    llm_model: str = "gpt-5-mini",
    extraction_model: str = "gpt-5-mini",
    schema: list[MetadataField] | None = None,
    data_dir: str = "./meta_rag_data",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    vector_store: VectorStore | None = None,
    relational_store: RelationalStore | None = None,
    prompts: PromptConfig | None = None,
)
```


| Parameter          | Default             | Description                                                              |
| -------------------- | --------------------- | -------------------------------------------------------------------------- |
| `llm_model`        | See `__init__.py`   | OpenAI model for query routing and answering                             |
| `extraction_model`  | See `__init__.py`   | OpenAI model used for metadata extraction during ingestion               |
| `schema`           | `None`              | List of `MetadataField`; auto-discovered on first ingest if omitted      |
| `data_dir`         | `"./meta_rag_data"` | Directory for ChromaDB and SQLite persistence                            |
| `chunk_size`       | `1000`              | Max characters per text chunk                                            |
| `chunk_overlap`    | `200`               | Character overlap between consecutive chunks                             |
| `vector_store`     | `None`              | Custom `VectorStore` (default created in `data_dir` if omitted)          |
| `relational_store` | `None`              | Custom `RelationalStore` (default created in `data_dir` if omitted)      |
| `prompts`          | `None`              | Custom `PromptConfig` for overriding system prompts                      |

### `ingest(path, on_progress=None) → dict`

Ingest a file path or list of paths. Skips unchanged documents (hash-based). Auto-discovers schema if none is set.

- `path` — `str` or `list[str]`
- `on_progress` — optional `(current: int, total: int) -> None` callback

Returns `{"new": int, "changed": int, "unchanged": int}`.

### `query(question, evolve=False, history=None, fallback=False) → str`

Ask a question. The LLM routes to semantic search, SQL, or both.

- `evolve=True` — after answering, check for schema gaps and add detected fields automatically
- `history` — list of prior `{"role": ..., "content": ...}` messages for multi-turn conversation
- `fallback=False` — when SQL fails (missing column or no rows), block fallback to semantic search to avoid incomplete answers. Set to `True` to allow the fallback with an incompleteness warning appended

Use `rag.last_history` to get the updated history after each call for follow-up questions. Use `rag.last_sql` to inspect the SQL that was generated (if any).

### `backfill(on_progress=None) → dict`

Extract values for all unpopulated fields from already-stored chunks. Prunes fields that remain entirely NULL after backfill.

- `on_progress` — optional `(current: int, total: int) -> None` callback

Returns `{"populated": [field names], "pruned": [field names]}`.

### `add_field(field: MetadataField) → None`

Add a new field to the live schema and the SQLite database. Call `backfill()` afterward to populate it from existing documents.
