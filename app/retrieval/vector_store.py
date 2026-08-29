"""Vector store.

Responsibility split (this matters for later phases):

* **VectorStore** = *storage + raw similarity math*. It knows about vectors and
  chunks. It does not know what an embedding model is or what a "question" is.
* **Retriever** (see `retriever.py`) = *orchestration*. It embeds the query, calls
  the store, and packages results into `RetrievedChunk`s with ranks.

Keeping them apart means we can later drop in FAISS / Chroma / pgvector by
implementing this one interface, and hybrid search (Phase 7+) becomes "a retriever
that queries two stores", not a rewrite.

Phase 1 implementation: an in-memory numpy matrix with brute-force cosine similarity.
Fine for a few thousand chunks; the abstraction is what lets us outgrow it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.models import Chunk

_EPS = 1e-12


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product == cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + _EPS)


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks alongside their vectors. `len(chunks) == len(embeddings)`."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Return up to `top_k` (chunk, similarity) pairs, best first."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored chunks."""


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        # shape (n_chunks, dim), rows L2-normalized; None until the first add()
        self._matrix = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        if not chunks:
            return

        vectors = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self._chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if self._matrix is None or not self._chunks:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        query = query / (np.linalg.norm(query) + _EPS)

        # (n_chunks, dim) @ (dim,) -> (n_chunks,) cosine similarities
        similarities = self._matrix @ query

        k = min(top_k, len(self._chunks))
        # argpartition gets the top-k unordered in O(n); then we sort just those k.
        top_idx = np.argpartition(-similarities, k - 1)[:k]
        top_idx = top_idx[np.argsort(-similarities[top_idx])]

        return [(self._chunks[i], float(similarities[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self._chunks)
