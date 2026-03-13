"""Example usage of meta-rag with manual schema, ingestion, and queries.

This script demonstrates the core workflow of meta-rag:
  1. Define a metadata schema with typed fields.
  2. Initialize MetaRAG with an LLM model and schema.
  3. Ingest a directory of documents (chunking + metadata extraction).
  4. Run qualitative (semantic) and quantitative (SQL) queries.

Requires: OPENAI_API_KEY environment variable set.

Usage:
    uv run python examples/example_usage.py [--verbose] [--test]
"""

import argparse
import os

from dotenv import load_dotenv

load_dotenv()

from meta_rag import MetaRAG, MetadataField


def main():
    parser = argparse.ArgumentParser(description="meta-rag example")
    parser.add_argument("--verbose", action="store_true", help="Print generated SQL")
    parser.add_argument("--test", action="store_true", help="Run demo queries before interactive mode")
    args = parser.parse_args()
    verbose = args.verbose
    # ------------------------------------------------------------------
    # 1. Define the metadata schema
    #    Each MetadataField describes a piece of structured information
    #    that the LLM will extract from every ingested document.
    # ------------------------------------------------------------------
    schema = [
        MetadataField(
            name="birthplace",
            type="text",
            description="Person's place of birth",
        ),
        MetadataField(
            name="occupation",
            type="text",
            description="Primary occupation or field",
        ),
        MetadataField(
            name="year_of_birth",
            type="integer",
            description="Year the person was born",
        ),
    ]

    # ------------------------------------------------------------------
    # 2. Initialize MetaRAG
    #    - llm_model:     which LLM to use for extraction and answering
    #    - schema:        the metadata fields defined above
    #    - data_dir:      where to store the vector store and SQL database
    #    - chunk_size:    maximum number of characters per text chunk
    #    - chunk_overlap: overlap between consecutive chunks
    # ------------------------------------------------------------------
    rag = MetaRAG(
        llm_model="gpt-4o",
        schema=schema,
        data_dir="./example_data",
        chunk_size=1000,
        chunk_overlap=200,
    )

    # ------------------------------------------------------------------
    # 3. Ingest documents
    #    Point MetaRAG at a directory of .txt files. It will:
    #      - Read and chunk each document
    #      - Use the LLM to extract metadata per the schema
    #      - Store chunks in a vector store and metadata in a SQL database
    # ------------------------------------------------------------------
    def _progress(current: int, total: int) -> None:
        print(f"  Ingesting file {current}/{total}...", end="\r")

    if not os.path.exists("./example_data/metadata.db"):
        print("Ingesting documents...")
        stats = rag.ingest("examples/documents/", on_progress=_progress)
        print(f"\nDone. new={stats['new']}  changed={stats['changed']}  unchanged={stats['unchanged']}")
        print(f"Schema fields: {[f.name for f in rag.schema.fields]}")
    else:
        print(f"Using existing data. Schema fields: {[f.name for f in rag.schema.fields]}")

    # ------------------------------------------------------------------
    # 4. (Optional) Run demo queries — only when --test is passed
    # ------------------------------------------------------------------
    if args.test:
        questions = [
            # Qualitative queries (semantic search)
            "Tell me about Newton's contributions to physics",
            "What did Marie Curie discover?",
            # Quantitative queries (SQL)
            "How many people in the dataset are from England?",
            "What is the most common occupation?",
            "What is the average birth year?",
            # Filtered
            "Who are the mathematicians in the collection?",
        ]

        for q in questions:
            print(f"\nQ: {q}")
            answer = rag.query(q)
            print(f"A: {answer}")
            if verbose and rag.last_sql:
                print(f"SQL: {rag.last_sql}")

    # ------------------------------------------------------------------
    # 5. Interactive query loop (with schema gap detection)
    #    Pass evolve=True so meta-rag detects missing schema fields and
    #    appends an informational message when a gap is found.
    #    After a gap is detected, call rag.backfill() to populate the
    #    new field from all already-stored document chunks.
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Interactive mode — type your questions (or 'quit' to exit)")
    print("  evolve=True is active: gaps trigger auto field addition.")
    print("  Type /backfill to populate newly added fields.")
    print("  Type /ingest to re-ingest documents from examples/documents/.")
    print("=" * 60)

    conversation_history: list[dict] = []

    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        if question.lower() in {"backfill", "/backfill"}:
            print("Running backfill...")

            def _backfill_progress(current: int, total: int) -> None:
                print(f"  Backfilling chunk {current}/{total}...", end="\r")

            result = rag.backfill(on_progress=_backfill_progress)
            print(f"\nBackfill complete. Populated: {result['populated']}  Pruned: {result['pruned']}")
            continue
        if question.lower() in {"ingest", "/ingest"}:
            print("Re-ingesting documents...")

            def _ingest_progress(current: int, total: int) -> None:
                print(f"  Ingesting file {current}/{total}...", end="\r")

            stats = rag.ingest("examples/documents/", on_progress=_ingest_progress)
            print(f"\nIngest complete. new={stats['new']}  changed={stats['changed']}  unchanged={stats['unchanged']}")
            continue
        answer = rag.query(question, evolve=True, history=conversation_history)
        conversation_history = rag.last_history
        print(f"A: {answer}")
        if verbose and rag.last_sql:
            print(f"SQL: {rag.last_sql}")


if __name__ == "__main__":
    main()
