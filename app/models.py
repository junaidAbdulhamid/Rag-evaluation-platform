"""Core domain models shared across the whole platform.

Every stage of the pipeline (ingestion -> retrieval -> generation) speaks in these
types instead of passing around raw dicts or tuples. Keeping them in one module means
later phases (evaluation, tracing, experiments) can import the same vocabulary without
creating circular dependencies.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A single source document loaded from disk, before any chunking."""

    document_id: str = Field(description="Stable id, unique per document (we use the filename stem).")
    filename: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """A slice of a document that will be embedded and stored as one vector."""

    chunk_id: str = Field(description="Stable id, unique per chunk, e.g. '<document_id>::chunk_3'.")
    document_id: str = Field(description="Which Document this chunk came from.")
    text: str
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, plus where it landed in the ranking."""

    chunk: Chunk
    score: float = Field(description="Similarity score from the vector store (higher = closer).")
    rank: int = Field(description="1-based position in the result list (1 = best match).")


class TokenUsage(BaseModel):
    """Token counts for a single model call. Cost estimation (Phase 11) builds on this."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class GeneratedAnswer(BaseModel):
    """The structured output of the generation stage."""

    answer: str
    model: str
    token_usage: Optional[TokenUsage] = None
    prompt: Optional[str] = Field(
        default=None,
        description="The exact user-message prompt sent to the model. Kept for tracing/debugging.",
    )


class RagResult(BaseModel):
    """Everything produced by one end-to-end pipeline run for one question."""

    question: str
    retrieved_chunks: list[RetrievedChunk]
    generated_answer: GeneratedAnswer


class IngestionResult(BaseModel):
    """Summary of an ingestion pass, handy for the CLI and for sanity checks."""

    document_count: int
    chunk_count: int
    document_ids: list[str]
