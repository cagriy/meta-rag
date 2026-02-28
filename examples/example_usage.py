"""Example usage of meta-rag with manual schema, ingestion, and queries.

This script demonstrates the core workflow of meta-rag:
  1. Define a metadata schema with typed fields.
  2. Initialize MetaRAG with an LLM model and schema.
  3. Ingest a directory of documents (chunking + metadata extraction).
  4. Run qualitative (semantic) and quantitative (SQL) queries.

Requires: OPENAI_API_KEY environment variable set.

Usage:
    uv run python examples/example_usage.py
"""

from meta_rag import MetaRAG, MetadataField


def main():
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
    print("Ingesting documents...")
    rag.ingest("examples/documents/")
    print(f"Done. Schema fields: {[f.name for f in rag.schema.fields]}")

    # ------------------------------------------------------------------
    # 4. Query the system
    #    meta-rag automatically routes each query:
    #      - Qualitative questions  -> semantic / vector search
    #      - Quantitative questions -> SQL over extracted metadata
    #      - Filtered questions     -> combined approach
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    main()
