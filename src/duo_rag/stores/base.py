from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from duo_rag.schema import MetadataField, MetadataSchema


@dataclass
class SearchResult:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict
    score: float


class VectorStore(ABC):
    @abstractmethod
    def initialize(self, schema: MetadataSchema) -> None: ...

    @abstractmethod
    def add(
        self,
        doc_id: str,
        chunk_id: str,
        text: str,
        embedding: list[float] | None,
        metadata: dict,
    ) -> None: ...

    @abstractmethod
    def search(
        self,
        query_text: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def clear_all(self) -> None: ...

    @abstractmethod
    def get_all_chunks(self) -> list[tuple[str, str, str]]: ...

    @abstractmethod
    def update_metadata(self, chunk_id: str, metadata: dict) -> None: ...

    @abstractmethod
    def get_document_chunks(self, doc_id: str) -> list[SearchResult]: ...


class RelationalStore(ABC):
    @abstractmethod
    def initialize(self, schema: MetadataSchema) -> None: ...

    @abstractmethod
    def insert(self, doc_id: str, metadata: dict) -> None: ...

    @abstractmethod
    def execute_sql(self, sql: str) -> list[dict]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def clear_all_data(self) -> None: ...

    @abstractmethod
    def get_schema_from_db(self) -> MetadataSchema | None: ...

    @abstractmethod
    def add_column(self, field: MetadataField) -> None: ...

    @abstractmethod
    def drop_column(self, field_name: str) -> None: ...

    @abstractmethod
    def get_empty_fields(self) -> list[str]: ...

    @abstractmethod
    def get_unpopulated_fields(self, schema: MetadataSchema) -> list[MetadataField]: ...

    @abstractmethod
    def update_metadata_field(self, doc_id: str, field_name: str, value: object) -> None: ...

    @abstractmethod
    def get_document_hash(self, doc_id: str) -> str | None: ...

    @abstractmethod
    def set_document_hash(self, doc_id: str, content_hash: str) -> None: ...
