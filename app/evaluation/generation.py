"""Generation evaluation: run the deterministic scorers and (optionally) the judge
over answers, then roll up per-question results into an aggregate.

Mirrors the shape of ``retrieval.py``:

* ``evaluate_generation_for_question`` - score one (example, answer) pair
* ``aggregate_generation_metrics``     - dataset-level means
* ``evaluate_generation``              - the driver, over (example, answer) cases

``evaluate_generation`` takes ``cases: Iterable[tuple[EvalExample, str]]`` - already
-produced answers - not a pipeline. Producing answers costs Anthropic tokens and is
the experiment runner's job (Phase 7); the evaluator only scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from pydantic import BaseModel

from app.evaluation.dataset import EvalExample
from app.evaluation.generation_metrics import (
    exact_match,
    is_abstention,
    number_coverage,
    token_f1,
    token_recall,
)
from app.evaluation.judge import GenerationJudge, GenerationJudgement


class DeterministicScores(BaseModel):
    exact_match: float
    token_f1: float
    token_recall: float
    number_coverage: Optional[float]  # None when the reference has no numbers
    abstention_expected: bool         # ground truth: should the system have declined?
    abstained: bool                   # did the generated answer actually decline?
    abstention_correct: bool          # abstained == abstention_expected


class QuestionGenerationResult(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    deterministic: DeterministicScores
    judgement: Optional[GenerationJudgement] = None


class AggregateGenerationMetrics(BaseModel):
    num_questions: int
    # deterministic means
    exact_match: float
    token_f1: float
    token_recall: float
    number_coverage: float       # over questions whose reference has numbers
    abstention_accuracy: float   # over all questions
    num_with_numbers: int
    # judge means (None when no judge was run)
    num_judged: int
    judge_correctness: Optional[float]
    judge_relevance: Optional[float]


class GenerationEvaluation(BaseModel):
    per_question: list[QuestionGenerationResult]
    aggregate: AggregateGenerationMetrics


def evaluate_generation_for_question(
    example: EvalExample,
    generated_answer: str,
    *,
    judge: Optional[GenerationJudge] = None,
) -> QuestionGenerationResult:
    abstained = is_abstention(generated_answer)
    deterministic = DeterministicScores(
        exact_match=exact_match(generated_answer, example.expected_answer),
        token_f1=token_f1(generated_answer, example.expected_answer),
        token_recall=token_recall(generated_answer, example.expected_answer),
        number_coverage=number_coverage(generated_answer, example.expected_answer),
        abstention_expected=example.is_unanswerable,
        abstained=abstained,
        abstention_correct=(abstained == example.is_unanswerable),
    )

    judgement: Optional[GenerationJudgement] = None
    if judge is not None:
        judgement = judge.judge(
            question=example.question,
            expected_answer=example.expected_answer,
            generated_answer=generated_answer,
        )

    return QuestionGenerationResult(
        question_id=example.id,
        question=example.question,
        expected_answer=example.expected_answer,
        generated_answer=generated_answer,
        deterministic=deterministic,
        judgement=judgement,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_generation_metrics(
    results: Sequence[QuestionGenerationResult],
) -> AggregateGenerationMetrics:
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    with_numbers = [
        r.deterministic.number_coverage
        for r in results
        if r.deterministic.number_coverage is not None
    ]
    judged = [r.judgement for r in results if r.judgement is not None]

    return AggregateGenerationMetrics(
        num_questions=len(results),
        exact_match=_mean([r.deterministic.exact_match for r in results]),
        token_f1=_mean([r.deterministic.token_f1 for r in results]),
        token_recall=_mean([r.deterministic.token_recall for r in results]),
        number_coverage=_mean(with_numbers),
        abstention_accuracy=_mean(
            [1.0 if r.deterministic.abstention_correct else 0.0 for r in results]
        ),
        num_with_numbers=len(with_numbers),
        num_judged=len(judged),
        judge_correctness=_mean([j.correctness for j in judged]) if judged else None,
        judge_relevance=_mean([j.relevance for j in judged]) if judged else None,
    )


def evaluate_generation(
    cases: Iterable[tuple[EvalExample, str]],
    *,
    judge: Optional[GenerationJudge] = None,
) -> GenerationEvaluation:
    per_question = [
        evaluate_generation_for_question(example, answer, judge=judge)
        for example, answer in cases
    ]
    return GenerationEvaluation(
        per_question=per_question,
        aggregate=aggregate_generation_metrics(per_question),
    )
