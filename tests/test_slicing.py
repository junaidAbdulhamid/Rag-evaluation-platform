"""Tests for per-slice metrics."""

from __future__ import annotations

from typing import Optional

import pytest

from app.evaluation.generation import DeterministicScores, QuestionGenerationResult
from app.evaluation.judge import GenerationJudgement
from app.evaluation.retrieval import QuestionRetrievalResult, RetrievalMetrics
from app.experiment.config import ExperimentConfig
from app.experiment.results import ExperimentResult, QuestionCost, QuestionExperimentResult
from app.experiment.slicing import slice_report, underperforming_slices


def q(
    qid: str,
    slices: list[str],
    *,
    recall: float = 1.0,
    correctness: float = 1.0,
    unanswerable: bool = False,
    latency: float = 1000.0,
    cost: float = 0.001,
) -> QuestionExperimentResult:
    relevant = [] if unanswerable else ["doc_a"]
    return QuestionExperimentResult(
        question_id=qid,
        question=f"question {qid}",
        slices=slices,
        retrieved_doc_ids=["doc_a"],
        generated_answer="b",
        retrieval=QuestionRetrievalResult(
            question_id=qid,
            retrieved_doc_ids=["doc_a"],
            relevant_doc_ids=relevant,
            metrics=RetrievalMetrics(
                k=4, hit_rate=1.0, precision=0.5, recall=(None if unanswerable else recall),
                reciprocal_rank=1.0, ndcg=1.0, num_relevant=len(relevant), num_retrieved_docs=1,
            ),
        ),
        generation=QuestionGenerationResult(
            question_id=qid, question="q", expected_answer="a", generated_answer="b",
            deterministic=DeterministicScores(
                exact_match=1.0, token_f1=1.0, token_recall=1.0, number_coverage=None,
                abstention_expected=unanswerable, abstained=unanswerable,
                abstention_correct=True,
            ),
            judgement=GenerationJudgement(
                correctness=correctness, relevance=1.0,
                correctness_reasoning="r", relevance_reasoning="r",
            ),
        ),
        latency_ms={"total": latency},
        cost=QuestionCost(total_usd=cost),
        estimated_cost_usd=cost,
    )


def experiment(questions) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp",
        config=ExperimentConfig(experiment_name="exp"),
        started_at="2026-09-01T00:00:00+00:00",
        finished_at="2026-09-01T00:05:00+00:00",
        num_questions=len(questions),
        num_errors=0,
        document_count=6,
        chunk_count=12,
        per_question=questions,
    )


def test_overall_and_per_slice_aggregates():
    report = slice_report(experiment([
        q("q1", ["simple_lookup", "numerical"], recall=1.0),
        q("q2", ["simple_lookup"], recall=1.0),
        q("q3", ["multi_document", "numerical"], recall=0.5),
    ]))

    labels = {s.label for s in report.slices}
    assert labels == {"simple_lookup", "numerical", "multi_document"}

    overall = report.overall
    assert overall.label == "overall" and overall.num_questions == 3
    assert overall.retrieval.recall == pytest.approx((1.0 + 1.0 + 0.5) / 3)

    by_label = {s.label: s for s in report.slices}
    assert by_label["simple_lookup"].retrieval.recall == 1.0     # q1, q2
    assert by_label["multi_document"].retrieval.recall == 0.5     # q3
    assert by_label["numerical"].retrieval.recall == pytest.approx(0.75)  # q1, q3
    assert by_label["numerical"].num_questions == 2


def test_slice_metric_accessor_and_latency_cost_means():
    report = slice_report(experiment([
        q("q1", ["policy"], latency=1000.0, cost=0.002),
        q("q2", ["policy"], latency=3000.0, cost=0.004),
    ]))
    policy = report.slices[0]
    assert policy.metric("recall") == 1.0
    assert policy.metric("correctness") == 1.0
    assert policy.metric("nonsense") is None
    assert policy.latency_total_ms == 2000.0
    assert policy.cost_per_query_usd == pytest.approx(0.003)


def test_unanswerable_only_slice_has_no_retrieval_metrics():
    report = slice_report(experiment([
        q("q1", ["unanswerable"], unanswerable=True),
        q("q2", ["unanswerable"], unanswerable=True),
    ]))
    unans = report.slices[0]
    assert unans.retrieval is None            # nothing scorable
    assert unans.generation is not None       # abstention is still measured
    assert unans.num_questions == 2


def test_underperforming_slices_flags_weak_query_types():
    report = slice_report(experiment([
        q("q1", ["simple_lookup"], recall=1.0),
        q("q2", ["simple_lookup"], recall=1.0),
        q("q3", ["simple_lookup"], recall=1.0),
        q("q4", ["multi_document"], recall=0.4),
        q("q5", ["multi_document"], recall=0.4),
    ]))

    weak = underperforming_slices(report, metric="recall", min_gap=0.05)
    assert [u.label for u in weak] == ["multi_document"]
    assert weak[0].gap < 0
    assert weak[0].slice_value == pytest.approx(0.4)

    # tighter slices are not flagged
    assert underperforming_slices(report, metric="recall", min_gap=0.9) == []
