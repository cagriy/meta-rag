from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    chunk_index: int = 0
    total_chunks: int = 1


class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str) -> list[Chunk]:
        """Split text by character count with overlap."""
        if len(text) <= self.chunk_size:
            return [Chunk(doc_id=doc_id, chunk_id=f"{doc_id}_chunk_0", text=text, chunk_index=0, total_chunks=1)]

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        i = 0
        idx = 0
        while i < len(text):
            chunk_text = text[i : i + self.chunk_size]
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_{idx}",
                    text=chunk_text,
                    chunk_index=idx,
                )
            )
            idx += 1
            i += step

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def chunk_file(self, path: str | Path) -> list[Chunk]:
        """Read a single file, use its stem as doc_id, call chunk_text."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        doc_id = path.stem
        return self.chunk_text(text, doc_id)

    def chunk_directory(self, dir_path: str | Path) -> list[Chunk]:
        """Read all .txt files in a directory (non-recursive), chunk each."""
        dir_path = Path(dir_path)
        chunks: list[Chunk] = []
        for file_path in sorted(dir_path.glob("*.txt")):
            chunks.extend(self.chunk_file(file_path))
        return chunks

    def chunk_paths(self, paths: list[str | Path]) -> list[Chunk]:
        """Accept a list of file/directory paths, chunk each appropriately."""
        chunks: list[Chunk] = []
        for path in paths:
            path = Path(path)
            if path.is_dir():
                chunks.extend(self.chunk_directory(path))
            else:
                chunks.extend(self.chunk_file(path))
        return chunks
