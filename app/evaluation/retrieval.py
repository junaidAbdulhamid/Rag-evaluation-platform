"""Retrieval evaluation: connect the raw metrics to the pipeline's output.

The pure functions in ``retrieval_metrics.py`` speak in document-id lists. This
module bridges from what the retriever actually returns (a ranked list of
``RetrievedChunk``) to those lists, computes per-question metrics, and rolls them
up into an aggregate.

## Chunk -> document reduction

Ground truth is labelled at the **document** level (``relevant_document_ids``), but
retrieval works at the **chunk** level, and one document can contribute several
chunks to the top-k. ``doc_ranking_from_chunks`` collapses that: it walks the chunks
in rank order and keeps the first occurrence of each document id. So three chunks
of ``refund_policy`` at ranks 1/2/5 become the single entry ``refund_policy`` at
effective rank 1. Every metric is then computed on this de-duplicated document
ranking - consistent unit throughout, and recall can never exceed 1.0.

## Unanswerable questions

Questions with no relevant documents (``is_unanswerable``) still get a per-question
record (recall / ndcg are ``None``), but they are **excluded from the aggregate** -
retrieval metrics are not meaningful when there is nothing to retrieve. Phase 13
analyses those questions through the "did the system correctly abstain?" lens
instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Optional

from pydantic import BaseModel

from app.evaluation.dataset import EvalExample
from app.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.models import RetrievedChunk

RetrieveFn = Callable[[str, int], list[RetrievedChunk]]


class RetrievalMetrics(BaseModel):
    """Metrics for one question at one cut-off ``k``."""

    k: int
    hit_rate: float
    precision: float
    recall: Optional[float]
    reciprocal_rank: float
    ndcg: Optional[float]
    num_relevant: int
    num_retrieved_docs: int


class QuestionRetrievalResult(BaseModel):
    """Per-question retrieval evaluation, kept for inspection and failure analysis."""

    question_id: str
    retrieved_doc_ids: list[str]  # de-duplicated, rank-ordered
    relevant_doc_ids: list[str]
    metrics: RetrievalMetrics

    @property
    def is_unanswerable(self) -> bool:
        return not self.relevant_doc_ids


class AggregateRetrievalMetrics(BaseModel):
    """Dataset-level means. Computed over answerable questions only."""

    k: int
    num_questions_total: int
    num_questions_scored: int  # answerable (>= 1 relevant doc)
    hit_rate: float
    precision: float
    recall: float
    mrr: float  # mean reciprocal_rank
    ndcg: float


class RetrievalEvaluation(BaseModel):
    """The full result of evaluating retrieval over a dataset."""

    k: int
    per_question: list[QuestionRetrievalResult]
    aggregate: AggregateRetrievalMetrics


def doc_ranking_from_chunks(retrieved: Sequence[RetrievedChunk]) -> list[str]:
    """Distinct source-document ids in retrieval-rank order (best rank wins)."""
    seen: set[str] = set()
    ordering: list[str] = []
    for item in sorted(retrieved, key=lambda rc: rc.rank):
        doc_id = item.chunk.document_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordering.append(doc_id)
    return ordering


def evaluate_retrieval_for_question(
    retrieved: Sequence[RetrievedChunk], example: EvalExample, k: int
) -> QuestionRetrievalResult:
    """Score one question's retrieved chunks against its ground-truth documents."""
    doc_ranking = doc_ranking_from_chunks(retrieved)
    relevant = example.relevant_document_ids

    metrics = RetrievalMetrics(
        k=k,
        hit_rate=hit_rate_at_k(doc_ranking, relevant, k),
        precision=precision_at_k(doc_ranking, relevant, k),
        recall=recall_at_k(doc_ranking, relevant, k),
        reciprocal_rank=reciprocal_rank(doc_ranking, relevant),
        ndcg=ndcg_at_k(doc_ranking, relevant, k),
        num_relevant=len(relevant),
        num_retrieved_docs=len(doc_ranking),
    )
    return QuestionRetrievalResult(
        question_id=example.id,
        retrieved_doc_ids=doc_ranking,
        relevant_doc_ids=list(relevant),
        metrics=metrics,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_retrieval_metrics(
    results: Sequence[QuestionRetrievalResult],
) -> AggregateRetrievalMetrics:
    """Roll per-question results into dataset-level means (answerable questions only)."""
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    k = results[0].metrics.k
    scored = [r for r in results if not r.is_unanswerable]

    return AggregateRetrievalMetrics(
        k=k,
        num_questions_total=len(results),
        num_questions_scored=len(scored),
        hit_rate=_mean([r.metrics.hit_rate for r in scored]),
        precision=_mean([r.metrics.precision for r in scored]),
        # recall / ndcg are guaranteed non-None for answerable questions
        recall=_mean([r.metrics.recall for r in scored if r.metrics.recall is not None]),
        mrr=_mean([r.metrics.reciprocal_rank for r in scored]),
        ndcg=_mean([r.metrics.ndcg for r in scored if r.metrics.ndcg is not None]),
    )


def evaluate_retrieval(
    dataset: Iterable[EvalExample], retrieve_fn: RetrieveFn, k: int
) -> RetrievalEvaluation:
    """Run ``retrieve_fn`` for every question in ``dataset`` and score the results.

    ``retrieve_fn`` is any callable ``(question, k) -> list[RetrievedChunk]`` - e.g.
    ``pipeline.retrieve``. Keeping it a parameter (rather than taking a pipeline)
    means Phase 7's experiment runner and the tests can plug in whatever they need.
    """
    per_question = [
        evaluate_retrieval_for_question(retrieve_fn(example.question, k), example, k)
        for example in dataset
    ]
    return RetrievalEvaluation(
        k=k,
        per_question=per_question,
        aggregate=aggregate_retrieval_metrics(per_question),
    )
