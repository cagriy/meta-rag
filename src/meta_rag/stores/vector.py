from __future__ import annotations

import chromadb

from meta_rag.schema import MetadataSchema
from meta_rag.stores.base import SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        self.collection = None
        self.schema: MetadataSchema | None = None

    def initialize(self, schema: MetadataSchema) -> None:
        self.schema = schema
        client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = client.get_or_create_collection("documents")

    def add(
        self,
        doc_id: str,
        chunk_id: str,
        text: str,
        embedding: list[float] | None,
        metadata: dict,
    ) -> None:
        # ChromaDB only accepts str, int, float, bool — filter out None values
        merged_metadata = {
            k: v
            for k, v in {**metadata, "doc_id": doc_id}.items()
            if v is not None
        }
        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            metadatas=[merged_metadata],
        )

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filters if filters else None,
        )

        search_results: list[SearchResult] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for chunk_id, text, distance, metadata in zip(
            ids, documents, distances, metadatas
        ):
            score = 1.0 / (1.0 + distance)
            doc_id = metadata.get("doc_id", chunk_id.rsplit("_", 1)[0])
            search_results.append(
                SearchResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                )
            )

        return search_results
