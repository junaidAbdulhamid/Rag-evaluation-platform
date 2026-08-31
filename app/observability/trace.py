"""The Trace model - a self-contained record of one RAG execution.

Mirrors the tree in the Phase 9 spec 1:1::

    Trace
    |-- question
    |-- retrieval   (query, embedding info, ranked chunks with scores)
    |-- generation  (model, prompt, answer, tokens, citations)
    |-- evaluation  (retrieval metrics, correctness, relevance, faithfulness, citations)
    `-- performance (per-stage latency, total tokens, cost)

A Trace duplicates some data that also lives in QuestionExperimentResult - that is
deliberate. A trace is meant to be readable and shareable on its own, without
needing the surrounding experiment.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.evaluation.citation import CitationEvaluationResult
from app.evaluation.faithfulness import FaithfulnessResult
from app.evaluation.generation import DeterministicScores
from app.evaluation.retrieval import RetrievalMetrics
from app.models import Citation, TokenUsage


class TracedChunk(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    text_preview: str


class RetrievalTrace(BaseModel):
    query: str
    top_k: int
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    chunks: list[TracedChunk] = Field(default_factory=list)
    embedding_ms: float = 0.0
    latency_ms: float = 0.0  # vector search only; embedding is separate


class GenerationTrace(BaseModel):
    model: str
    prompt: str
    answer: str
    token_usage: Optional[TokenUsage] = None
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: float = 0.0


class EvaluationTrace(BaseModel):
    retrieval_metrics: Optional[RetrievalMetrics] = None
    correctness: Optional[float] = None
    relevance: Optional[float] = None
    deterministic: Optional[DeterministicScores] = None
    faithfulness: Optional[FaithfulnessResult] = None
    citation: Optional[CitationEvaluationResult] = None
    latency_ms: float = 0.0


class PerformanceTrace(BaseModel):
    embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0


class Trace(BaseModel):
    trace_id: str
    question: str
    question_id: Optional[str] = None
    started_at: str

    retrieval: Optional[RetrievalTrace] = None
    generation: Optional[GenerationTrace] = None
    evaluation: Optional[EvaluationTrace] = None
    performance: PerformanceTrace = Field(default_factory=PerformanceTrace)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors
