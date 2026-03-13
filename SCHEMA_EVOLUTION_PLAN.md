# Schema Evolution Plan

## Context

When a user asks "How many people worked on relativity?", the system can't answer because `research_interests` is not in the schema. We need a way to detect this gap and offer to add the field. The challenge is distinguishing relevant gaps from irrelevant questions ("How many dogs do they have?") — we don't want schema inflation.

**Chosen approach: optimistic add + post-ingestion pruning.**

1. The LLM detects if a question exposes a schema gap and proposes a new field (lightweight check — no sampling)
2. User confirms → field is added immediately
3. User re-ingests → extraction runs for all documents including the new field
4. After ingestion, `ingest()` automatically prunes fields where every value is NULL — they had no data in the corpus and are removed
5. User gets a clear report: which fields were populated vs pruned

This is simpler and more reliable than upfront extractability heuristics — it uses actual ingestion evidence.

A secondary fix: **field descriptions are lost** when schema is recovered from DB via `PRAGMA table_info`. We add a `_schema_fields` table to persist them.

---

## What Changes

### 1. `src/meta_rag/stores/relational.py`

**a) Add `_schema_fields` table** for description persistence:
```sql
CREATE TABLE IF NOT EXISTS _schema_fields (
    name TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
)
```
- `initialize(schema)`: insert all fields into `_schema_fields` after creating `documents_metadata`
- `get_schema_from_db()`: load descriptions from `_schema_fields` and return full `MetadataField` objects

**b) `add_column(field: MetadataField) -> None`**:
- `ALTER TABLE documents_metadata ADD COLUMN {name} {SQL_TYPE}`
- `INSERT OR REPLACE INTO _schema_fields (name, type, description) VALUES (?, ?, ?)`

**c) `drop_column(field_name: str) -> None`**:
- `ALTER TABLE documents_metadata DROP COLUMN {field_name}` (SQLite ≥3.35, available in Python 3.11+)
- `DELETE FROM _schema_fields WHERE name = ?`

**d) `get_empty_fields() -> list[str]`**:
- For each non-system column in `documents_metadata`, check `SELECT COUNT(*) WHERE {col} IS NOT NULL`
- Return list of field names where count == 0

### 2. `src/meta_rag/schema.py`

**a) `SchemaEvolutionResult` dataclass**:
```python
@dataclass
class SchemaEvolutionResult:
    gap_detected: bool
    reasoning: str
    proposed_field: MetadataField | None
```

**b) `MetadataSchema.add_field(field: MetadataField) -> None`** — appends to `self.fields`.

**c) `MetadataSchema.remove_field(field_name: str) -> None`** — removes by name from `self.fields`.

### 3. `src/meta_rag/__init__.py`

**`evolve_schema(question: str) -> SchemaEvolutionResult`**:
- Sends current schema field list + user question to LLM
- LLM determines: does this question need a new field for SQL aggregation?
- Returns `SchemaEvolutionResult`

LLM prompt:
```
You are evaluating whether a question requires a new metadata field.

Current schema fields: [list]

User question: "{question}"

Does answering this question via SQL aggregation require a field not in the current schema?
Only propose a field if it's a structured fact that could realistically appear in documents of this type.

Return JSON:
{
  "gap_detected": bool,
  "reasoning": "...",
  "proposed_field": {"name": ..., "type": ..., "description": ...} | null
}
```

**`add_field(field: MetadataField) -> None`**:
- `self.relational_store.add_column(field)` → ALTER TABLE + persist description
- `self.schema.add_field(field)` → update in-memory schema

**`ingest()` — add post-ingestion pruning**:
After the pipeline completes, call `_prune_empty_fields()`.

**`_prune_empty_fields() -> list[str]` (private)**:
1. `empty = self.relational_store.get_empty_fields()`
2. For each empty field: `self.relational_store.drop_column(name)` + `self.schema.remove_field(name)`
3. Return list of pruned field names (caller prints the report)

**New export:** `SchemaEvolutionResult`

### 4. `examples/example_usage.py`

In the interactive loop, after each answer:
```python
result = rag.evolve_schema(question)
if result.gap_detected:
    print(f"\n[Schema gap] {result.reasoning}")
    f = result.proposed_field
    print(f"Proposed: {f.name} ({f.type}) — {f.description}")
    confirm = input("Add this field? (y/n): ").strip().lower()
    if confirm == "y":
        rag.add_field(f)
        print(f"Field '{f.name}' added. Re-ingesting to populate it...")
        rag.ingest("examples/documents/")
        print("Done. Try your question again.")
```

After `ingest()`, the pruning runs automatically. The example should print what was pruned (ingest returns/prints a summary).

---

## Critical Files

| File | Change |
|------|--------|
| `src/meta_rag/stores/relational.py` | `_schema_fields` table, `add_column()`, `drop_column()`, `get_empty_fields()`, update `initialize()` + `get_schema_from_db()` |
| `src/meta_rag/schema.py` | `SchemaEvolutionResult`, `MetadataSchema.add_field()`, `MetadataSchema.remove_field()` |
| `src/meta_rag/__init__.py` | `evolve_schema()`, `add_field()`, `_prune_empty_fields()`, post-ingest pruning in `ingest()`, export `SchemaEvolutionResult` |
| `examples/example_usage.py` | Gap detection + confirmation + re-ingest in interactive loop |

---

## Verification

1. `uv run pytest tests/ -v` — all 62 existing tests still pass
2. New tests:
   - `evolve_schema("How many people worked on relativity?")` (mocked LLM) → `gap_detected=True`
   - `evolve_schema("How many dogs do they have?")` (mocked LLM) → `gap_detected=False`
   - `add_field()` → `add_column()` called, schema updated in memory
   - `get_empty_fields()` → returns field names with all NULLs
   - `drop_column()` → column removed, `get_schema_from_db()` no longer returns it
   - `_prune_empty_fields()` → empty fields removed from both DB and in-memory schema
   - `get_schema_from_db()` → descriptions preserved across re-instantiation
3. Manual: ask "How many dogs do they have?" → LLM says no gap. Ask "How many people worked on relativity?" → gap detected, confirm, re-ingest → `pets`-like fields pruned, `research_interests` kept if any documents had data.
