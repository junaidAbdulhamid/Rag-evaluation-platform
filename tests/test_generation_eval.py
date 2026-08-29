"""Tests for the generation-evaluation glue: per-question scoring, judge wiring,
and aggregation."""

import json

import pytest

from app.evaluation.dataset import EvalExample
from app.evaluation.generation import (
    aggregate_generation_metrics,
    evaluate_generation,
    evaluate_generation_for_question,
)
from app.evaluation.judge import LLMGenerationJudge
from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY
from tests.fakes import FakeTextLLM

JUDGE_JSON = json.dumps(
    {
        "correctness": 0.9,
        "relevance": 0.9,
        "correctness_reasoning": "ok",
        "relevance_reasoning": "ok",
    }
)


def answerable(qid="q1") -> EvalExample:
    return EvalExample(
        id=qid,
        question="How many days for a refund?",
        expected_answer="Customers have 30 days to request a refund.",
        relevant_document_ids=["refund_policy"],
    )


def unanswerable(qid="qU") -> EvalExample:
    return EvalExample(
        id=qid,
        question="Do you price match?",
        expected_answer=INSUFFICIENT_CONTEXT_REPLY,
        relevant_document_ids=[],
        slices=["unanswerable"],
    )


# --- evaluate_generation_for_question -------------------------------------------------
def test_good_answer_scores_high_deterministically():
    r = evaluate_generation_for_question(answerable(), "Customers have 30 days to request a refund.")
    d = r.deterministic
    assert d.exact_match == 1.0
    assert d.token_f1 == 1.0
    assert d.number_coverage == 1.0        # "30" present
    assert d.abstention_expected is False
    assert d.abstained is False
    assert d.abstention_correct is True
    assert r.judgement is None             # no judge passed


def test_wrong_answer_scores_low():
    r = evaluate_generation_for_question(answerable(), "Refunds take about a year, I think.")
    assert r.deterministic.exact_match == 0.0
    assert r.deterministic.number_coverage == 0.0  # "30" missing


def test_unanswerable_with_correct_abstention():
    r = evaluate_generation_for_question(unanswerable(), INSUFFICIENT_CONTEXT_REPLY)
    assert r.deterministic.abstention_expected is True
    assert r.deterministic.abstained is True
    assert r.deterministic.abstention_correct is True


def test_unanswerable_but_model_hallucinated():
    r = evaluate_generation_for_question(unanswerable(), "Yes, we price match any competitor.")
    assert r.deterministic.abstention_expected is True
    assert r.deterministic.abstained is False
    assert r.deterministic.abstention_correct is False


def test_judge_is_invoked_when_provided():
    judge = LLMGenerationJudge(FakeTextLLM(JUDGE_JSON))
    r = evaluate_generation_for_question(answerable(), "You have 30 days.", judge=judge)
    assert r.judgement is not None
    assert r.judgement.correctness == 0.9


# --- aggregate_generation_metrics --------------------------------------------------
def test_aggregate_means_and_counts():
    results = [
        evaluate_generation_for_question(answerable("q1"), "Customers have 30 days to request a refund."),
        evaluate_generation_for_question(answerable("q2"), "wrong answer entirely"),
        evaluate_generation_for_question(unanswerable("qU"), INSUFFICIENT_CONTEXT_REPLY),
    ]
    agg = aggregate_generation_metrics(results)

    assert agg.num_questions == 3
    # q1 matches its reference exactly; qU's answer == the abstention sentence == its
    # reference, so that is also an exact match. q2 is wrong. -> 2/3
    assert agg.exact_match == pytest.approx(2 / 3)
    assert agg.abstention_accuracy == 1.0              # q1/q2 didn't abstain (correct), qU did (correct)
    assert agg.num_with_numbers == 2                   # q1 & q2 share a reference with "30" in it
    assert agg.number_coverage == pytest.approx(0.5)   # q1 has "30", q2 does not
    assert agg.num_judged == 0
    assert agg.judge_correctness is None


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_generation_metrics([])


# --- evaluate_generation (driver) -----------------------------------------------------
def test_evaluate_generation_end_to_end_with_fake_judge():
    judge = LLMGenerationJudge(FakeTextLLM(JUDGE_JSON))
    cases = [
        (answerable("q1"), "Customers have 30 days to request a refund."),
        (unanswerable("qU"), INSUFFICIENT_CONTEXT_REPLY),
    ]
    evaluation = evaluate_generation(cases, judge=judge)

    assert [r.question_id for r in evaluation.per_question] == ["q1", "qU"]
    assert evaluation.aggregate.num_judged == 2
    assert evaluation.aggregate.judge_correctness == pytest.approx(0.9)
    assert evaluation.aggregate.abstention_accuracy == 1.0
