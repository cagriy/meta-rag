from __future__ import annotations

import chromadb

from meta_rag.schema import MetadataSchema
from meta_rag.stores.base import SearchResult, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir
        self.collection = None
        self._client = None
        self.schema: MetadataSchema | None = None

    def initialize(self, schema: MetadataSchema) -> None:
        self.schema = schema
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self._client.get_or_create_collection("documents")

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

    def delete_document(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def clear_all(self) -> None:
        self._client.delete_collection("documents")
        self.collection = self._client.create_collection("documents")

    def get_all_chunks(self) -> list[tuple[str, str, str]]:
        result = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            doc_id = metadata.get("doc_id", chunk_id.rsplit("_", 1)[0])
            chunks.append((doc_id, chunk_id, text))
        return chunks

    def get_document_chunks(self, doc_id: str) -> list[SearchResult]:
        result = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        chunks: list[SearchResult] = []
        for chunk_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"]
        ):
            chunks.append(
                SearchResult(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=0.0,
                )
            )
        chunks.sort(key=lambda c: c.metadata.get("chunk_index", 0))
        return chunks

    def update_metadata(self, chunk_id: str, metadata: dict) -> None:
        result = self.collection.get(ids=[chunk_id], include=["metadatas"])
        if not result["ids"]:
            return
        current = result["metadatas"][0] or {}
        merged = {**current, **metadata}
        # ChromaDB does not accept None values
        merged = {k: v for k, v in merged.items() if v is not None}
        self.collection.update(ids=[chunk_id], metadatas=[merged])
