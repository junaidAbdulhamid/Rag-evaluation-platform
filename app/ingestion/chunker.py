"""Chunking.

A chunker splits a `Document` into smaller `Chunk`s. Chunk size and overlap are two
of the most important levers on retrieval quality, so they are explicit constructor
parameters, not hidden constants. Phase 7's experiment runner sweeps these values.

Phase 1 uses a simple **character-based sliding window**:

    |<--------- chunk_size --------->|
    |                     |<-overlap->|
    0                   step        chunk_size
                          |<--------- chunk_size --------->|

`step = chunk_size - chunk_overlap` is how far the window advances each iteration.
Overlap keeps sentences that straddle a boundary from being lost to both chunks.

Character windows are crude (they can cut mid-word) but they are trivial to reason
about and to unit-test. A smarter sentence/markdown-aware splitter can replace this
class later without touching the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.models import Chunk, Document


@dataclass
class TextChunker:
    chunk_size: int = 500
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    @property
    def _step(self) -> int:
        return self.chunk_size - self.chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split one document. Returns [] for empty/whitespace-only text."""
        text = document.text
        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            piece = text[start : start + self.chunk_size]
            if piece.strip():  # skip windows that are only whitespace
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}::chunk_{index}",
                        document_id=document.document_id,
                        text=piece,
                        metadata={
                            **document.metadata,
                            "chunk_index": index,
                            "char_start": start,
                            "char_end": start + len(piece),
                        },
                    )
                )
                index += 1
            start += self._step

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk many documents and flatten the result into a single list."""
        return [chunk for document in documents for chunk in self.chunk_document(document)]
