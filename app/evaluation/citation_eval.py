"""Dataset-level citation evaluation - the glue around ``CitationEvaluator``.

Same shape as the other ``*_eval`` modules. Driver consumes
``(question_id, answer_text, retrieved_chunks)`` tuples - it needs the answer text
(with markers) and the chunks (to resolve markers and read chunk text), but no
golden answer.

Aggregates report both a **macro** mean (per-question metric, averaged over the
questions where it is defined) and **micro** totals (links summed across the whole
dataset), because a single wordy answer with many citations should not dominate the
macro average.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from pydantic import BaseModel

from app.evaluation.citation import CitationEvaluationResult, CitationEvaluator
from app.models import RetrievedChunk


class QuestionCitationResult(BaseModel):
    question_id: str
    answer: str
    result: CitationEvaluationResult

    @property
    def has_claims(self) -> bool:
        return self.result.num_claims > 0


class AggregateCitationMetrics(BaseModel):
    num_questions: int
    num_scored: int  # questions with at least one factual claim
    # macro means (over questions where the metric is defined)
    citation_completeness: float
    citation_precision: float
    citation_correctness: float
    citation_hallucination_rate: float
    # micro totals
    total_links: int
    total_supported_links: int
    total_hallucinated_links: int


class CitationEvaluation(BaseModel):
    per_question: list[QuestionCitationResult]
    aggregate: AggregateCitationMetrics


def evaluate_citations_for_question(
    question_id: str,
    answer: str,
    retrieved: Sequence[RetrievedChunk],
    evaluator: CitationEvaluator,
) -> QuestionCitationResult:
    result = evaluator.evaluate(answer=answer, retrieved=retrieved)
    return QuestionCitationResult(question_id=question_id, answer=answer, result=result)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _defined(results: Sequence[QuestionCitationResult], attr: str) -> list[float]:
    return [
        getattr(r.result, attr)
        for r in results
        if getattr(r.result, attr) is not None
    ]


def aggregate_citation_metrics(
    results: Sequence[QuestionCitationResult],
) -> AggregateCitationMetrics:
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    scored = [r for r in results if r.has_claims]
    return AggregateCitationMetrics(
        num_questions=len(results),
        num_scored=len(scored),
        citation_completeness=_mean(_defined(results, "citation_completeness")),
        citation_precision=_mean(_defined(results, "citation_precision")),
        citation_correctness=_mean(_defined(results, "citation_correctness")),
        citation_hallucination_rate=_mean(_defined(results, "citation_hallucination_rate")),
        total_links=sum(r.result.num_citation_links for r in results),
        total_supported_links=sum(r.result.num_supported_links for r in results),
        total_hallucinated_links=sum(r.result.num_hallucinated_links for r in results),
    )


def evaluate_citations(
    cases: Iterable[tuple[str, str, Sequence[RetrievedChunk]]],
    evaluator: CitationEvaluator,
) -> CitationEvaluation:
    per_question = [
        evaluate_citations_for_question(question_id, answer, retrieved, evaluator)
        for question_id, answer, retrieved in cases
    ]
    return CitationEvaluation(
        per_question=per_question,
        aggregate=aggregate_citation_metrics(per_question),
    )
