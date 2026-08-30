"""Dataset-level faithfulness evaluation - the glue around ``FaithfulnessEvaluator``.

Mirrors ``retrieval.py`` / ``generation.py``:

* ``evaluate_faithfulness_for_question`` - one answer + its retrieved context
* ``aggregate_faithfulness_metrics``      - dataset-level roll-up
* ``evaluate_faithfulness``               - the driver, over (id, RagResult) cases

Faithfulness needs no golden answer, so the driver consumes ``RagResult`` objects
(question + retrieved chunks + generated answer), not ``EvalExample``.

Two aggregate numbers, on purpose:
* ``faithfulness`` - **macro** average: mean of the per-question scores (each
  question counts once, regardless of how many claims it made). Skips questions
  with no factual claims.
* ``claim_support_rate`` - **micro** average: total supported claims / total claims
  across the whole dataset (a wordy 6-claim answer counts 6x). They diverge when
  answer lengths vary, and seeing both is the point.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from pydantic import BaseModel

from app.evaluation.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from app.models import RagResult


class QuestionFaithfulnessResult(BaseModel):
    question_id: str
    answer: str
    result: FaithfulnessResult

    @property
    def has_claims(self) -> bool:
        return self.result.num_claims > 0


class AggregateFaithfulnessMetrics(BaseModel):
    num_questions: int
    num_scored: int          # questions that produced at least one factual claim
    faithfulness: float      # macro: mean of per-question scores
    total_claims: int
    total_supported: int
    claim_support_rate: float  # micro: total_supported / total_claims


class FaithfulnessEvaluation(BaseModel):
    per_question: list[QuestionFaithfulnessResult]
    aggregate: AggregateFaithfulnessMetrics


def evaluate_faithfulness_for_question(
    question_id: str,
    answer: str,
    retrieved,
    evaluator: FaithfulnessEvaluator,
) -> QuestionFaithfulnessResult:
    result = evaluator.evaluate(answer=answer, retrieved=retrieved)
    return QuestionFaithfulnessResult(question_id=question_id, answer=answer, result=result)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_faithfulness_metrics(
    results: Sequence[QuestionFaithfulnessResult],
) -> AggregateFaithfulnessMetrics:
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    scored = [r for r in results if r.has_claims]
    total_claims = sum(r.result.num_claims for r in results)
    total_supported = sum(r.result.num_supported for r in results)

    return AggregateFaithfulnessMetrics(
        num_questions=len(results),
        num_scored=len(scored),
        faithfulness=_mean([r.result.score for r in scored if r.result.score is not None]),
        total_claims=total_claims,
        total_supported=total_supported,
        claim_support_rate=(total_supported / total_claims) if total_claims else 0.0,
    )


def evaluate_faithfulness(
    cases: Iterable[tuple[str, RagResult]],
    evaluator: FaithfulnessEvaluator,
) -> FaithfulnessEvaluation:
    per_question = [
        evaluate_faithfulness_for_question(
            question_id,
            rag_result.generated_answer.answer,
            rag_result.retrieved_chunks,
            evaluator,
        )
        for question_id, rag_result in cases
    ]
    return FaithfulnessEvaluation(
        per_question=per_question,
        aggregate=aggregate_faithfulness_metrics(per_question),
    )
