"""The shape of a completed experiment.

Per-question results embed the full Phase 3-6 ``Question*Result`` objects (not just
their scalar scores) so that:
* the dataset-level aggregates can reuse the existing ``aggregate_*`` functions, and
* Phase 9's tracing has everything it needs without re-running anything.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.evaluation.citation_eval import AggregateCitationMetrics, QuestionCitationResult
from app.evaluation.faithfulness_eval import (
    AggregateFaithfulnessMetrics,
    QuestionFaithfulnessResult,
)
from app.evaluation.generation import AggregateGenerationMetrics, QuestionGenerationResult
from app.evaluation.retrieval import AggregateRetrievalMetrics, QuestionRetrievalResult
from app.experiment.config import ExperimentConfig
from app.models import TokenUsage
from app.observability.trace import Trace


class LatencySummary(BaseModel):
    """Mean stage latency across all questions, in milliseconds. Phase 10 adds p50/p95."""

    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    evaluation_ms: float = 0.0
    total_ms: float = 0.0


class ExperimentError(BaseModel):
    question_id: str
    stage: str
    message: str


class QuestionExperimentResult(BaseModel):
    question_id: str
    question: str
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    generated_answer: str = ""

    retrieval: Optional[QuestionRetrievalResult] = None
    generation: Optional[QuestionGenerationResult] = None
    faithfulness: Optional[QuestionFaithfulnessResult] = None
    citation: Optional[QuestionCitationResult] = None

    latency_ms: dict = Field(default_factory=dict)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class ExperimentResult(BaseModel):
    experiment_id: str
    config: ExperimentConfig
    started_at: str
    finished_at: str

    num_questions: int
    num_errors: int
    document_count: int
    chunk_count: int

    per_question: list[QuestionExperimentResult]
    traces: list[Trace] = Field(default_factory=list)

    retrieval: Optional[AggregateRetrievalMetrics] = None
    generation: Optional[AggregateGenerationMetrics] = None
    faithfulness: Optional[AggregateFaithfulnessMetrics] = None
    citation: Optional[AggregateCitationMetrics] = None

    latency: LatencySummary = Field(default_factory=LatencySummary)
    total_token_usage: TokenUsage = Field(default_factory=TokenUsage)
    estimated_cost_usd: float = 0.0
    errors: list[ExperimentError] = Field(default_factory=list)
