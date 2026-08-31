"""TraceRecorder - accumulate the pieces of one RAG execution, then build a Trace.

Two ways to feed it latency:
* pass a shared ``latency`` dict at construction (the experiment runner already times
  its stages with ``record_ms`` into such a dict), or
* use ``recorder.stage("retrieval")`` as a context manager (the standalone path).

Everything else is fed through ``record_*`` calls after each stage runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.evaluation.citation import CitationEvaluationResult
from app.evaluation.faithfulness import FaithfulnessResult
from app.evaluation.generation import DeterministicScores
from app.evaluation.judge import GenerationJudgement
from app.evaluation.retrieval import RetrievalMetrics
from app.models import Citation, RetrievedChunk, TokenUsage
from app.observability.timing import record_ms
from app.observability.trace import (
    EvaluationTrace,
    GenerationTrace,
    PerformanceTrace,
    RetrievalTrace,
    Trace,
    TracedChunk,
)

_PREVIEW_CHARS = 200


def _preview(text: str) -> str:
    flat = " ".join(text.split())
    return flat[:_PREVIEW_CHARS] + ("..." if len(flat) > _PREVIEW_CHARS else "")


class TraceRecorder:
    def __init__(
        self,
        question: str,
        question_id: Optional[str] = None,
        latency: Optional[dict] = None,
    ) -> None:
        self.trace_id = uuid4().hex
        self.question = question
        self.question_id = question_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._latency: dict = latency if latency is not None else {}
        self._retrieval: Optional[RetrievalTrace] = None
        self._generation: Optional[GenerationTrace] = None
        self._evaluation: Optional[EvaluationTrace] = None
        self._errors: list[str] = []

    @contextmanager
    def stage(self, name: str):
        """Time a block into ``self._latency[name]`` (standalone tracing path)."""
        with record_ms(self._latency, name):
            yield

    def record_error(self, message: str) -> None:
        self._errors.append(message)

    def record_retrieval(
        self,
        *,
        query: str,
        retrieved: Sequence[RetrievedChunk],
        top_k: int,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        self._retrieval = RetrievalTrace(
            query=query,
            top_k=top_k,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            latency_ms=self._latency.get("retrieval", 0.0),
            chunks=[
                TracedChunk(
                    rank=rc.rank,
                    score=rc.score,
                    chunk_id=rc.chunk.chunk_id,
                    document_id=rc.chunk.document_id,
                    text_preview=_preview(rc.chunk.text),
                )
                for rc in retrieved
            ],
        )

    def record_generation(
        self,
        *,
        model: str,
        prompt: str,
        answer: str,
        token_usage: Optional[TokenUsage] = None,
        citations: Optional[Sequence[Citation]] = None,
    ) -> None:
        self._generation = GenerationTrace(
            model=model,
            prompt=prompt,
            answer=answer,
            token_usage=token_usage,
            citations=list(citations or []),
            latency_ms=self._latency.get("generation", 0.0),
        )

    def record_evaluation(
        self,
        *,
        retrieval_metrics: Optional[RetrievalMetrics] = None,
        judgement: Optional[GenerationJudgement] = None,
        deterministic: Optional[DeterministicScores] = None,
        faithfulness: Optional[FaithfulnessResult] = None,
        citation: Optional[CitationEvaluationResult] = None,
    ) -> None:
        self._evaluation = EvaluationTrace(
            retrieval_metrics=retrieval_metrics,
            correctness=judgement.correctness if judgement else None,
            relevance=judgement.relevance if judgement else None,
            deterministic=deterministic,
            faithfulness=faithfulness,
            citation=citation,
            latency_ms=self._latency.get("evaluation", 0.0),
        )

    def build(
        self, *, token_usage: TokenUsage, estimated_cost_usd: float
    ) -> Trace:
        return Trace(
            trace_id=self.trace_id,
            question=self.question,
            question_id=self.question_id,
            started_at=self.started_at,
            retrieval=self._retrieval,
            generation=self._generation,
            evaluation=self._evaluation,
            performance=PerformanceTrace(
                retrieval_ms=self._latency.get("retrieval", 0.0),
                generation_ms=self._latency.get("generation", 0.0),
                evaluation_ms=self._latency.get("evaluation", 0.0),
                total_ms=self._latency.get("total", 0.0),
                token_usage=token_usage,
                estimated_cost_usd=estimated_cost_usd,
            ),
            errors=self._errors,
        )
