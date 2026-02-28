from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meta_rag.schema import MetadataSchema


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


class RelationalStore(ABC):
    @abstractmethod
    def initialize(self, schema: MetadataSchema) -> None: ...

    @abstractmethod
    def insert(self, doc_id: str, chunk_id: str, metadata: dict) -> None: ...

    @abstractmethod
    def execute_sql(self, sql: str) -> list[dict]: ...
