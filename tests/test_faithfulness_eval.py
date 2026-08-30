"""Tests for the dataset-level faithfulness glue: per-question, macro vs micro
aggregation, and the driver."""

import pytest

from app.evaluation.faithfulness import Claim, FaithfulnessResult
from app.evaluation.faithfulness_eval import (
    aggregate_faithfulness_metrics,
    evaluate_faithfulness,
    evaluate_faithfulness_for_question,
)
from app.models import GeneratedAnswer, RagResult
from tests.fakes import FakeFaithfulnessEvaluator


def claim(text: str, supported: bool) -> Claim:
    return Claim(text=text, supported=supported, reason="r")


def result(*claims: Claim) -> FaithfulnessResult:
    return FaithfulnessResult.from_claims(list(claims))


def rag(answer: str) -> RagResult:
    return RagResult(
        question="q?",
        retrieved_chunks=[],
        generated_answer=GeneratedAnswer(answer=answer, model="m"),
    )


def test_per_question_delegates_to_evaluator():
    evaluator = FakeFaithfulnessEvaluator({"a": result(claim("x", True), claim("y", False))})
    r = evaluate_faithfulness_for_question("q1", "a", [], evaluator)

    assert r.question_id == "q1"
    assert r.result.score == 0.5
    assert r.has_claims is True


def test_aggregate_macro_vs_micro():
    # q1: 1/1 supported. q2: 1/3 supported. q3: no claims (skipped by macro).
    results = [
        evaluate_faithfulness_for_question("q1", "a1", [], FakeFaithfulnessEvaluator(
            {"a1": result(claim("x", True))})),
        evaluate_faithfulness_for_question("q2", "a2", [], FakeFaithfulnessEvaluator(
            {"a2": result(claim("x", True), claim("y", False), claim("z", False))})),
        evaluate_faithfulness_for_question("q3", "a3", [], FakeFaithfulnessEvaluator(
            {"a3": FaithfulnessResult.no_claims()})),
    ]
    agg = aggregate_faithfulness_metrics(results)

    assert agg.num_questions == 3
    assert agg.num_scored == 2                          # q3 has no claims
    assert agg.faithfulness == pytest.approx((1.0 + 1 / 3) / 2)   # macro: mean of q1,q2 scores
    assert agg.total_claims == 4
    assert agg.total_supported == 2
    assert agg.claim_support_rate == pytest.approx(0.5)           # micro: 2/4


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_faithfulness_metrics([])


def test_driver_end_to_end():
    evaluator = FakeFaithfulnessEvaluator(
        {
            "supported answer": result(claim("x", True), claim("y", True)),
            "made up answer": result(claim("z", False)),
        }
    )
    cases = [("q1", rag("supported answer")), ("q2", rag("made up answer"))]

    evaluation = evaluate_faithfulness(cases, evaluator)

    assert [r.question_id for r in evaluation.per_question] == ["q1", "q2"]
    assert evaluation.aggregate.faithfulness == pytest.approx((1.0 + 0.0) / 2)
    assert evaluation.aggregate.claim_support_rate == pytest.approx(2 / 3)
