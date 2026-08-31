"""Retriever.

Turns a natural-language question into a ranked list of `RetrievedChunk`s by:

    1. embedding the question with the same provider used for the chunks
    2. asking the vector store for the nearest vectors
    3. attaching a 1-based rank to each hit

`BaseRetriever` exists so that Phase 7's advanced strategies (hybrid search, query
rewriting, reranking, metadata filtering) are all just alternative implementations of
`retrieve()` - the pipeline and the evaluators never need to know which one is wired
in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingestion.embeddings import EmbeddingProvider
from app.models import RetrievedChunk
from app.observability.latency import measure
from app.retrieval.vector_store import VectorStore


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to `top_k` chunks ranked most-relevant first."""


class DenseRetriever(BaseRetriever):
    """Single-vector (a.k.a. "dense") similarity search."""

    def __init__(self, embeddings: EmbeddingProvider, vector_store: VectorStore) -> None:
        self._embeddings = embeddings
        self._vector_store = vector_store

    def retrieve(self, question: str, top_k: int) -> list[RetrievedChunk]:
        with measure("embedding"):
            query_vector = self._embeddings.embed_text(question)
        with measure("retrieval"):
            hits = self._vector_store.search(query_vector, top_k)
        return [
            RetrievedChunk(chunk=chunk, score=score, rank=position)
            for position, (chunk, score) in enumerate(hits, start=1)
        ]
