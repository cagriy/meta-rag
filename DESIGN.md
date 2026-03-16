# meta-rag: Design Document

## 1. Problem Statement

### What RAG Does Today

Retrieval-Augmented Generation (RAG) systems retrieve relevant documents from a knowledge base and pass them to a Large Language Model (LLM) to generate answers. The standard pipeline is:

```
User question → Embed question → Vector similarity search → Top-k documents → LLM → Answer
```

This works well for **qualitative queries** — questions where the answer lives inside one or a few documents:

- "Tell me about Newton's early work in optics"
- "What did Einstein say about time dilation?"

### Where RAG Fails

RAG fundamentally cannot answer **aggregation queries** — questions that require computation across many documents:

- "How many people in the dataset are from England?"
- "What is the most common birthplace?"
- "What percentage of scientists were born before 1900?"

Even if birthplace metadata has been extracted and stored, the RAG pipeline has no mechanism to count, sum, or group across the entire corpus. Vector similarity search returns the top-k most similar documents — it cannot perform `COUNT(*)`, `AVG()`, or `GROUP BY`.

### What AutoRAG Does

[AutoRAG](https://github.com/Marker-Inc-Korea/AutoRAG) (Marker Inc. Korea) is an AutoML tool that automatically optimizes RAG pipelines. It tries different combinations of query expansion strategies, retrieval methods, reranking models, and generation models, then evaluates each combination against your data and selects the best pipeline. However, AutoRAG's optimization space is limited to retrieval-based pipelines — it has no concept of structured queries, metadata aggregation, or SQL.

### What meta-rag Solves

**meta-rag** is a Python library that extends RAG with structured query capabilities. It:

1. **Extracts metadata** from documents during ingestion (automatically or from a user-defined schema)
2. **Stores data in two backends** — a vector database for semantic search, and a relational database for structured queries
3. **Routes queries intelligently** — the LLM decides whether a question needs semantic search, SQL aggregation, or both
4. **Handles everything in a single LLM call** — no secondary text-to-SQL model needed

The result: a single library that answers both "tell me about Newton" and "how many scientists were born in England" correctly.

---

## 2. Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    Consumer Application                  │
│                                                          │
│   from meta_rag import MetaRAG                          │
│   rag = MetaRAG(llm_model="gpt-5-mini")                    │
│   rag.ingest("docs/")                                   │
│   answer = rag.query("How many people from England?")   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                     meta-rag Library                     │
│                                                          │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Schema   │  │ Ingestion Pipeline│  │ Query Pipeline│  │
│  │ (auto or  │  │                  │  │               │  │
│  │  manual)  │──│ chunk → extract  │  │ route → exec  │  │
│  │          │  │  → dual-store    │  │  → synthesize │  │
│  └──────────┘  └────────┬─────────┘  └───────┬───────┘  │
│                         │                     │          │
│              ┌──────────┴──────────┐         │          │
│              ▼                     ▼         ▼          │
│  ┌─────────────────┐  ┌─────────────────────────┐      │
│  │   Vector Store   │  │   Relational Store       │      │
│  │   (ChromaDB)     │  │   (SQLite)               │      │
│  │                  │  │                          │      │
│  │  semantic search │  │  COUNT, SUM, AVG,        │      │
│  │  + metadata      │  │  GROUP BY, etc.          │      │
│  │  filtering       │  │                          │      │
│  └─────────────────┘  └─────────────────────────┘      │
│                                                          │
│  Both embedded — no external servers required            │
└─────────────────────────────────────────────────────────┘
```

### Why Two Databases?

| Capability | ChromaDB (Vector Store) | SQLite (Relational Store) |
|---|---|---|
| Semantic similarity search | Yes | No |
| Metadata filtering | Yes | Yes |
| COUNT(*) | No | Yes |
| SUM(), AVG(), MIN(), MAX() | No | Yes |
| GROUP BY | No | Yes |
| Complex JOINs | No | Yes |
| Embedded (no server) | Yes | Yes (stdlib) |

ChromaDB excels at "find documents similar to this query" but cannot count, sum, or group. SQLite excels at aggregation but cannot do semantic search. meta-rag uses both, routing each query to the appropriate backend.

Both are **embedded** — ChromaDB runs in-process with data stored in a local directory, and SQLite is part of Python's standard library writing to a single `.db` file. No servers, no infrastructure, no configuration.

### Future: Server-Backed Backends

For production deployments with large datasets or multiple frontends accessing the same data, the embedded stores can be swapped for server-backed alternatives via the abstract store interfaces:

```python
# Development / small scale (default — embedded, zero config)
rag = MetaRAG(llm_model="gpt-5-mini")

# Production / multi-frontend (future)
rag = MetaRAG(
    llm_model="gpt-5-mini",
    vector_store=PgVectorStore(url="postgresql://..."),
    relational_store=PgRelationalStore(url="postgresql://..."),
)
```

PostgreSQL with pgvector could implement both interfaces in a single database.

---

## 3. Package Structure

```
meta-rag/
├── pyproject.toml                  # uv-managed, package metadata & dependencies
├── examples/
│   └── example_usage.py            # Minimal working example
├── src/
│   └── meta_rag/
│       ├── __init__.py             # MetaRAG facade (public API surface)
│       ├── schema.py               # MetadataField, MetadataSchema
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # IngestionPipeline orchestrator
│       │   ├── chunker.py          # Document chunking
│       │   └── extractor.py        # LLM-based metadata extraction
│       ├── query/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # QueryPipeline orchestrator
│       │   ├── tools.py            # ToolBuilder — generates LLM tool definitions
│       │   └── executor.py         # ToolExecutor — runs SQL or vector search
│       └── stores/
│           ├── __init__.py
│           ├── base.py             # Abstract VectorStore + RelationalStore interfaces
│           ├── vector.py           # ChromaDB implementation
│           └── relational.py       # SQLite implementation
└── tests/
    ├── test_schema.py
    ├── test_ingestion.py
    └── test_query.py
```

---

## 4. Metadata Schema

The schema defines what structured fields to extract from documents. It can be auto-discovered or manually specified.

### Schema Lifecycle

The schema only needs to be defined **once** — before the very first ingestion. After that, the library reads it back from the database automatically. The user never has to re-specify it.

```
First ingestion (no data exists):
  Schema (via Option A or B) ──→ Create DB tables ──→ Extract & store ──→ Done

Every run after (data exists):
  Read schema from DB ──→ Ready to query (or ingest more documents)
```

**Initialization logic:**
1. Check if the database already exists in `data_dir`
2. **If yes** → read schema from the database (`PRAGMA table_info(documents_metadata)`), no schema needed from the user
3. **If no** → schema must be determined before ingestion can begin (via one of the two options below)

### Two Options for Initial Schema

Before the first ingestion, the schema can come from one of two places:

**Option A — Auto-discovery (default):** The library samples a few documents and asks the LLM to propose a schema. The user provides nothing — the library figures it out.

```python
rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_data")
rag.ingest("path/to/biographies/")

# Internally at the start of ingest():
# 1. No existing DB found → need a schema
# 2. No schema provided → sample first 5-10 documents
# 3. Send to LLM: "What structured fields can be extracted from these?"
# 4. LLM responds: [{"name": "birthplace", "type": "text", ...}, ...]
# 5. Create DB tables from discovered schema
# 6. Proceed with extraction and storage for ALL documents
```

**Option B — Manual specification:** The user tells the library exactly what fields to extract.

```python
from meta_rag import MetaRAG, MetadataField

rag = MetaRAG(
    llm_model="gpt-5-mini",
    data_dir="./my_data",
    schema=[
        MetadataField(name="birthplace", type="text", description="Person's place of birth"),
        MetadataField(name="occupation", type="text", description="Primary occupation"),
        MetadataField(name="year_of_birth", type="integer", description="Year the person was born"),
    ],
)
rag.ingest("./documents/")
```

**Either way, the result is identical** — a schema exists, DB tables are created, and the ingestion pipeline extracts those fields from every chunk and writes to both stores. The two options only differ in *who decides the schema* (LLM vs user).

### Subsequent Sessions

After the first ingestion, the schema lives in the database. The user just points to the same `data_dir`:

```python
# No schema needed — read from the existing database
rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_data")
rag.query("How many people are from England?")
rag.query("What is the most common occupation?")

# Can also ingest more documents — schema is already known
rag.ingest("./more_documents/")
```

### What the Schema Drives

The schema is the foundation of the entire system. From it, meta-rag automatically generates three things:

**1. Extraction prompts** (used during ingestion to tell the LLM what to pull from each chunk):
```
Extract the following fields from this document chunk.
Return JSON with these fields (use null if not found):

- birthplace (text): Person's place of birth
- occupation (text): Primary occupation
- year_of_birth (integer): Year the person was born
```

**2. SQL table DDL** (creates the relational store table):
```sql
CREATE TABLE documents_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    birthplace TEXT,
    occupation TEXT,
    year_of_birth INTEGER
);
```

**3. LLM tool definitions** (tells the query LLM what SQL fields exist):
```json
{
    "name": "run_sql",
    "description": "Run a SQL query for counting, summing, or comparing. Table: documents_metadata(id, doc_id, chunk_id, birthplace TEXT, occupation TEXT, year_of_birth INTEGER). Only SELECT.",
    "parameters": {
        "sql": {"type": "string", "description": "A valid SQL SELECT query"}
    }
}
```

---

## 5. Store Interfaces

Abstract interfaces that decouple the library from specific database implementations. Any application can provide custom implementations to swap backends.

### VectorStore (stores/base.py)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict
    score: float


class VectorStore(ABC):
    @abstractmethod
    def initialize(self, schema: "MetadataSchema") -> None:
        """Set up the store (create collection, etc.)."""
        ...

    @abstractmethod
    def add(self, doc_id: str, chunk_id: str, text: str,
            embedding: list[float], metadata: dict) -> None:
        """Store a document chunk with its embedding and metadata."""
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 10,
               filters: dict | None = None) -> list[SearchResult]:
        """Semantic similarity search with optional metadata filtering."""
        ...
```

### RelationalStore (stores/base.py)

```python
class RelationalStore(ABC):
    @abstractmethod
    def initialize(self, schema: "MetadataSchema") -> None:
        """Create the metadata table from schema."""
        ...

    @abstractmethod
    def insert(self, doc_id: str, chunk_id: str, metadata: dict) -> None:
        """Insert a metadata row."""
        ...

    @abstractmethod
    def execute_sql(self, sql: str) -> list[dict]:
        """Execute a read-only SQL query and return results as dicts."""
        ...
```

### Default Implementations

- `ChromaVectorStore` (stores/vector.py) — wraps `chromadb` with an embedded persistent client
- `SQLiteRelationalStore` (stores/relational.py) — wraps `sqlite3`, creates the table from schema DDL

### Custom Implementations

Applications can provide their own store implementations:

```python
from meta_rag.stores.base import VectorStore, RelationalStore

class MyPgVectorStore(VectorStore):
    def __init__(self, connection_url: str):
        self.url = connection_url
    # ... implement abstract methods

rag = MetaRAG(
    llm_model="gpt-5-mini",
    vector_store=MyPgVectorStore("postgresql://localhost/mydb"),
)
```

---

## 6. Ingestion Pipeline

### Flow

```
Documents (files or directory)
        │
        ▼
   ┌─────────┐
   │ Chunker  │  Split documents into chunks
   └────┬─────┘
        │
        ▼
  ┌───────────┐   (only if no schema provided)
  │  Schema   │   Analyze sample docs → LLM proposes schema
  │ Discovery │
  └─────┬─────┘
        │
        ▼
  ┌───────────┐
  │ Extractor │   For each chunk, LLM extracts metadata per schema
  └─────┬─────┘
        │
        ├──────────────────────┐
        ▼                      ▼
  ┌───────────┐         ┌───────────┐
  │ ChromaDB  │         │  SQLite   │
  │           │         │           │
  │ text      │         │ metadata  │
  │ embedding │         │ doc_id    │
  │ metadata  │         │ chunk_id  │
  └───────────┘         └───────────┘
```

### Chunking

Documents are split into manageable chunks before processing:

```python
# Default: split by character count with overlap
rag = MetaRAG(llm_model="gpt-5-mini", chunk_size=1000, chunk_overlap=200)
```

### Metadata Extraction

For each chunk, the LLM is prompted to extract the schema fields:

1. Builds a prompt from the schema (field names, types, descriptions)
2. Includes the chunk text
3. Asks the LLM to return structured JSON
4. Validates the response matches the schema types
5. Returns the extracted metadata dict

Example LLM interaction during extraction:

```
System: You are a metadata extraction assistant. Extract the following fields
from the given text. Return valid JSON. Use null for fields not found.

Fields:
- birthplace (text): Person's place of birth
- occupation (text): Primary occupation
- year_of_birth (integer): Year the person was born

User: "Sir Isaac Newton was born on 25 December 1642 in Woolsthorpe-by-Colsterworth,
a hamlet in the county of Lincolnshire, England. He was a mathematician, physicist,
and astronomer."
LLM responds:
{"birthplace": "Woolsthorpe-by-Colsterworth, England", "occupation": "mathematician, physicist, astronomer", "year_of_birth": 1642}
```

### Dual-Write

After extraction, each chunk is written to both stores in a single transaction-like operation:

```python
# Pseudocode from IngestionPipeline
for chunk in chunks:
    metadata = extractor.extract(chunk.text, schema)
    embedding = embed(chunk.text)

    vector_store.add(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        embedding=embedding,
        metadata=metadata,
    )

    relational_store.insert(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        metadata=metadata,
    )
```

---

## 7. Query Pipeline

The query pipeline is the core innovation of meta-rag. It uses the LLM's native tool-calling capability to route queries to the appropriate backend.

### How It Works

```
User question
      │
      ▼
  QueryPipeline
      │
      ▼
  LLM call #1: "Answer this question using the available tools"
      │          (system prompt + tool definitions built from schema)
      │
      ▼
  LLM picks a tool:
   ┌──────────────┬───────────────────────────┐
   │              │                           │
   ▼              ▼                           ▼
run_sql      semantic_search          (no tool — LLM
   │              │                    answers directly)
   ▼              ▼
SQLite        ChromaDB
   │              │
   ▼              ▼
result         results
   │              │
   └──────┬───────┘
          ▼
  LLM call #2: "Here are the tool results. Now answer the user's question."
          │
          ▼
  Natural language answer returned to application
```

### Tool Definitions

The `ToolBuilder` generates two tool definitions from the schema. These are passed to the LLM via the OpenAI function-calling API so the LLM knows what tools are available:

```python
# semantic_search tool — for qualitative questions
{
    "type": "function",
    "function": {
        "name": "semantic_search",
        "description": (
            "Search documents by meaning and content. "
            "Use for qualitative questions like 'tell me about', 'what did X say', "
            "'explain', 'describe', 'find documents about'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters to narrow results",
                    "properties": {
                        "birthplace": {"type": "string"},
                        "occupation": {"type": "string"},
                        "year_of_birth": {"type": "integer"}
                    }
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)"
                }
            },
            "required": ["query"]
        }
    }
}

# run_sql tool — for aggregation and quantitative questions
{
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Run a SQL query for counting, summing, averaging, comparing, "
            "or listing distinct values. Use for quantitative questions like "
            "'how many', 'what percentage', 'most common', 'average', 'total'. "
            "Table: documents_metadata("
            "id INTEGER PRIMARY KEY, "
            "doc_id TEXT, "
            "chunk_id TEXT, "
            "birthplace TEXT, "
            "occupation TEXT, "
            "year_of_birth INTEGER"
            "). Only SELECT queries are allowed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQL SELECT query"
                }
            },
            "required": ["sql"]
        }
    }
}
```

The key insight: the tool descriptions contain enough guidance ("use for questions like 'how many'...") for the LLM to route correctly, and the `run_sql` description embeds the full table schema so the LLM can generate valid SQL in a single call.

### Query Routing Examples

| User Question | LLM Picks | What Happens |
|---|---|---|
| "Tell me about Newton's early life" | `semantic_search(query="Newton early life")` | ChromaDB similarity search returns relevant chunks |
| "How many people are from England?" | `run_sql(sql="SELECT COUNT(*) FROM documents_metadata WHERE birthplace LIKE '%England%'")` | SQLite executes the count |
| "What is the most common occupation?" | `run_sql(sql="SELECT occupation, COUNT(*) as cnt FROM documents_metadata GROUP BY occupation ORDER BY cnt DESC LIMIT 1")` | SQLite groups and counts |
| "Who from France wrote about mathematics?" | `semantic_search(query="mathematics", filters={"birthplace": "France"})` | ChromaDB filtered search |
| "Average birth year of physicists?" | `run_sql(sql="SELECT AVG(year_of_birth) FROM documents_metadata WHERE occupation LIKE '%physicist%'")` | SQLite computes average |

### SQL Safety

The `ToolExecutor` validates SQL before execution:

- **Whitelist**: Only `SELECT` statements are allowed
- **Blacklist**: Rejects any query containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `EXEC`
- **Read-only connection**: SQLite connection opened with `uri=True` and `?mode=ro` flag
- **Timeout**: Query execution timeout to prevent runaway queries

```python
def execute_sql(self, sql: str) -> list[dict]:
    normalized = sql.strip().upper()
    if not normalized.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "EXEC"]
    for keyword in forbidden:
        if keyword in normalized:
            raise ValueError(f"Forbidden SQL keyword: {keyword}")

    # Execute with read-only connection
    conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
    cursor = conn.execute(sql)
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

---

## 8. Consumer Integration Guide

### Installation

```bash
uv add meta-rag
```

This installs meta-rag and its dependencies (`openai`, `chromadb`). SQLite is included in Python's standard library.

### Minimal Example — First Run (Auto-Discovery)

```python
from meta_rag import MetaRAG

# First run — no existing data, no schema provided
# Auto-discovery will analyze documents and propose a schema
rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_data")

# Ingest documents — schema is discovered, then metadata extracted
rag.ingest("./documents/")

# Query — the library handles routing internally
print(rag.query("How many people are from England?"))
# → "There are 47 people from England in the dataset."

print(rag.query("Tell me about Newton's contributions to physics"))
# → "Isaac Newton made foundational contributions to physics including..."

print(rag.query("What is the most common birthplace?"))
# → "The most common birthplace is London, with 23 people."
```

### Minimal Example — Subsequent Runs (Schema From Database)

```python
from meta_rag import MetaRAG

# Later session — schema is loaded from the existing database automatically
rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_data")

# No need to re-ingest or re-specify schema — just query
print(rag.query("What percentage of people are physicists?"))
```

### First Run With Manual Schema

```python
from meta_rag import MetaRAG, MetadataField

# First run — user provides schema explicitly
rag = MetaRAG(
    llm_model="gpt-5-mini",
    schema=[
        MetadataField(name="birthplace", type="text", description="Person's place of birth"),
        MetadataField(name="occupation", type="text", description="Primary occupation"),
        MetadataField(name="year_of_birth", type="integer", description="Year the person was born"),
    ],
    data_dir="./my_rag_data",
    chunk_size=1000,
    chunk_overlap=200,
)

rag.ingest(["./bios/newton.txt", "./bios/einstein.txt", "./bios/curie.txt"])

answer = rag.query("What is the average birth year of all scientists?")
print(answer)

# On next run, just:
# rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./my_rag_data")
# rag.query(...)  — schema is read from the database
```

### Integration with a Web Application

```python
from fastapi import FastAPI
from meta_rag import MetaRAG, MetadataField

app = FastAPI()

# Initialize once at startup
rag = MetaRAG(
    llm_model="gpt-5-mini",
    schema=[
        MetadataField(name="birthplace", type="text", description="Person's place of birth"),
        MetadataField(name="occupation", type="text", description="Primary occupation"),
        MetadataField(name="year_of_birth", type="integer", description="Year the person was born"),
    ],
    data_dir="./rag_data",
)


@app.post("/ingest")
async def ingest_documents(directory: str):
    """Ingest documents from a directory."""
    rag.ingest(directory)
    return {"status": "ok"}


@app.post("/query")
async def query(question: str):
    """Ask a question — meta-rag handles routing automatically."""
    answer = rag.query(question)
    return {"answer": answer}


# The consumer doesn't need to know whether the question
# will be answered via SQL or semantic search — meta-rag decides.
#
# POST /query {"question": "How many people are from England?"}
#   → internally routes to SQL: SELECT COUNT(*) ...
#   → {"answer": "There are 47 people from England."}
#
# POST /query {"question": "Tell me about Newton"}
#   → internally routes to ChromaDB semantic search
#   → {"answer": "Isaac Newton was born in 1642..."}
```

### Integration with a CLI Application

```python
from meta_rag import MetaRAG

def main():
    rag = MetaRAG(llm_model="gpt-5-mini", data_dir="./knowledge_base")

    # One-time ingestion
    print("Ingesting documents...")
    rag.ingest("./documents/")
    print(f"Ingested. Schema: {rag.schema}")

    # Interactive query loop
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer = rag.query(question)
        print(f"\n{answer}")

if __name__ == "__main__":
    main()
```

### Accessing the Discovered Schema

After ingestion with auto-discovery, the consumer can inspect what schema was discovered:

```python
rag = MetaRAG(llm_model="gpt-5-mini")
rag.ingest("./documents/")

# See what fields were auto-discovered
for field in rag.schema.fields:
    print(f"  {field.name} ({field.type}): {field.description}")

# Output:
#   birthplace (text): Person's place of birth
#   occupation (text): Primary occupation
#   year_of_birth (integer): Year the person was born
```

### Using Custom Store Backends

For production deployments, consumers can provide their own store implementations:

```python
from meta_rag import MetaRAG, MetadataField
from meta_rag.stores.base import VectorStore, RelationalStore


class MyPgVectorStore(VectorStore):
    """Custom pgvector implementation for production use."""

    def __init__(self, connection_url: str):
        self.url = connection_url

    def initialize(self, schema):
        # Create pgvector table with embedding column
        ...

    def add(self, doc_id, chunk_id, text, embedding, metadata):
        # INSERT INTO documents (doc_id, chunk_id, text, embedding, metadata) ...
        ...

    def search(self, query_embedding, top_k=10, filters=None):
        # SELECT ... ORDER BY embedding <=> query_embedding LIMIT top_k
        ...


class MyPgRelationalStore(RelationalStore):
    """Custom PostgreSQL implementation for production use."""

    def __init__(self, connection_url: str):
        self.url = connection_url

    def initialize(self, schema):
        # CREATE TABLE documents_metadata (...)
        ...

    def insert(self, doc_id, chunk_id, metadata):
        # INSERT INTO documents_metadata ...
        ...

    def execute_sql(self, sql):
        # Execute with read-only transaction
        ...


# Use custom backends — same MetaRAG API, different infrastructure
rag = MetaRAG(
    llm_model="gpt-5-mini",
    vector_store=MyPgVectorStore("postgresql://prod-server/mydb"),
    relational_store=MyPgRelationalStore("postgresql://prod-server/mydb"),
)
```

---

## 9. Dependencies

| Package | Purpose | Required? |
|---|---|---|
| `openai` | LLM provider for extraction, routing, and generation | Yes |
| `chromadb` | Embedded vector store for semantic search | Yes (default backend) |
| `sqlite3` | Embedded relational store for SQL aggregation | Yes (Python stdlib, zero install) |

### Embedding Model

ChromaDB includes its own default embedding model (based on sentence-transformers). For v1, meta-rag will use ChromaDB's built-in embeddings to keep setup simple. OpenAI embeddings can be added as an option later.

---

## 10. Implementation Order

| Step | Module | Description |
|---|---|---|
| 1 | `schema.py` | MetadataField + MetadataSchema — foundation everything depends on |
| 2 | `stores/base.py` | Abstract VectorStore + RelationalStore interfaces |
| 3 | `stores/relational.py` | SQLite implementation |
| 4 | `stores/vector.py` | ChromaDB implementation |
| 5 | `ingestion/chunker.py` | Document chunking |
| 6 | `ingestion/extractor.py` | LLM-based metadata extraction |
| 7 | `ingestion/pipeline.py` | Ingestion orchestrator with optional auto-discovery |
| 8 | `query/tools.py` | Dynamic tool definition builder |
| 9 | `query/executor.py` | Tool execution (SQL + vector search) |
| 10 | `query/pipeline.py` | Query orchestrator |
| 11 | `__init__.py` | MetaRAG facade wiring everything together |
| 12 | `examples/example_usage.py` | Working example |
| 13 | Tests | Unit + integration tests |

---

## 11. Testing Strategy

### Unit Tests

- **Schema**: Verify `MetadataSchema` generates correct SQL DDL, extraction prompts, and tool definitions from field definitions
- **SQL safety**: Verify `ToolExecutor` rejects non-SELECT queries, dangerous keywords, and malformed SQL
- **Chunking**: Verify documents are split correctly with overlap

### Integration Tests

Ingest a small set of test documents (e.g., 10 short biographies), then verify:

```python
# Aggregation via SQL
assert "47" in rag.query("How many people are from England?")

# Semantic search via ChromaDB
assert "Newton" in rag.query("Tell me about the physicist from Woolsthorpe")

# GROUP BY via SQL
assert rag.query("What is the most common birthplace?")  # should return a valid answer

# Filtered semantic search
assert rag.query("What did scientists from France work on?")  # should use filters + search
```

### Manual Testing

Run against a real document set and inspect:
- Which tool the LLM selects for each query
- Whether the generated SQL is valid and correct
- Whether semantic search results are relevant
