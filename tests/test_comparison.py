"""Tests for experiment comparison: deltas, config diff, tradeoffs."""

from __future__ import annotations

import pytest

from app.evaluation.faithfulness_eval import AggregateFaithfulnessMetrics
from app.evaluation.generation import AggregateGenerationMetrics
from app.evaluation.retrieval import AggregateRetrievalMetrics
from app.experiment.comparison import _delta, compare_experiments, format_delta
from app.experiment.config import ExperimentConfig
from app.experiment.results import CostBreakdown, ExperimentResult, LatencySummary
from app.observability.latency import LatencyReport


def make_result(
    eid: str,
    *,
    chunk_size: int = 500,
    top_k: int = 4,
    recall: float = 0.9,
    correctness: float = 0.9,
    faithfulness: float | None = 0.9,
    total_latency_ms: float = 2000.0,
    cost_per_query: float = 0.001,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=eid,
        config=ExperimentConfig(experiment_name=eid, chunk_size=chunk_size, top_k=top_k),
        started_at="2026-09-01T00:00:00+00:00",
        finished_at="2026-09-01T00:05:00+00:00",
        num_questions=10,
        num_errors=0,
        document_count=6,
        chunk_count=12,
        per_question=[],
        retrieval=AggregateRetrievalMetrics(
            k=top_k, num_questions_total=10, num_questions_scored=10,
            hit_rate=1.0, precision=0.5, recall=recall, mrr=1.0, ndcg=1.0,
        ),
        generation=AggregateGenerationMetrics(
            num_questions=10, exact_match=0.7, token_f1=0.9, token_recall=1.0,
            number_coverage=1.0, abstention_accuracy=1.0, num_with_numbers=6,
            num_judged=10, judge_correctness=correctness, judge_relevance=1.0,
        ),
        faithfulness=(
            AggregateFaithfulnessMetrics(
                num_questions=10, num_scored=10, faithfulness=faithfulness,
                total_claims=20, total_supported=int(20 * faithfulness),
                claim_support_rate=faithfulness,
            )
            if faithfulness is not None
            else None
        ),
        latency=LatencySummary(total_ms=total_latency_ms, generation_ms=800.0, evaluation_ms=1000.0),
        latency_report=LatencyReport.from_question_timings([{"total": total_latency_ms}]),
        cost=CostBreakdown(total_usd=cost_per_query * 10, cost_per_query_usd=cost_per_query),
    )


# --- _delta ---------------------------------------------------------------------------
def test_delta_improved_for_higher_is_better():
    d = _delta(0.80, 0.88, higher_is_better=True)
    assert d.absolute == pytest.approx(0.08)
    assert d.percent == pytest.approx(10.0)
    assert d.direction == "improved"


def test_delta_regressed_for_higher_is_better():
    assert _delta(0.90, 0.80, higher_is_better=True).direction == "regressed"


def test_delta_lower_is_better_flips_direction():
    # latency going up is a regression
    assert _delta(2000.0, 2320.0, higher_is_better=False).direction == "regressed"
    # cost going down is an improvement
    assert _delta(0.005, 0.001, higher_is_better=False).direction == "improved"


def test_delta_small_change_is_neutral():
    assert _delta(1.0, 1.002, higher_is_better=True).direction == "neutral"  # 0.2% < 0.5%


def test_delta_zero_baseline_has_no_percent():
    d = _delta(0.0, 0.05, higher_is_better=False)
    assert d.percent is None and d.direction == "regressed"


# --- compare_experiments -------------------------------------------------------------
def test_needs_two_experiments():
    with pytest.raises(ValueError):
        compare_experiments([make_result("a")])


def test_baseline_has_no_deltas_others_do():
    report = compare_experiments([make_result("base"), make_result("b", recall=0.95)])
    recall_mc = next(m for m in report.metrics if m.key == "recall")

    assert report.baseline_id == "base"
    assert recall_mc.deltas["base"] is None
    assert recall_mc.deltas["b"].direction == "improved"
    assert recall_mc.values == {"base": 0.9, "b": 0.95}


def test_config_diff_lists_only_changed_fields():
    report = compare_experiments([
        make_result("a", chunk_size=500, top_k=4),
        make_result("b", chunk_size=1000, top_k=4),
    ])
    fields = {cd.field for cd in report.config_diff}
    assert "chunk_size" in fields
    assert "top_k" not in fields          # unchanged
    assert "experiment_name" not in fields  # always differs, excluded


def test_missing_metric_yields_none_delta_without_crashing():
    report = compare_experiments([
        make_result("a", faithfulness=0.9),
        make_result("b", faithfulness=None),
    ])
    faith_mc = next(m for m in report.metrics if m.key == "faithfulness")
    assert faith_mc.values["b"] is None
    assert faith_mc.deltas["b"] is None


# --- tradeoffs ---------------------------------------------------------------------------
def test_quality_up_but_slower_and_costlier_is_flagged_as_a_tradeoff():
    report = compare_experiments([
        make_result("base", recall=0.85, total_latency_ms=2000.0, cost_per_query=0.001),
        make_result("bigger", recall=0.95, total_latency_ms=2400.0, cost_per_query=0.004),
    ])
    t = next(t for t in report.tradeoffs if t.experiment_id == "bigger")

    assert t.summary == "higher quality, but slower and/or more expensive"
    assert any("Recall@K" in g for g in t.gains)
    assert any("Latency" in loss for loss in t.losses)
    assert any("Cost" in loss for loss in t.losses)


def test_strictly_better_summary():
    report = compare_experiments([
        make_result("base", recall=0.80, correctness=0.80),
        make_result("better", recall=0.95, correctness=0.95),
    ])
    t = report.tradeoffs[0]
    assert t.summary == "strictly better than the baseline on measured metrics"
    assert not t.losses


def test_faster_cheaper_but_worse_quality():
    report = compare_experiments([
        make_result("base", recall=0.95, correctness=0.95, total_latency_ms=3000.0, cost_per_query=0.01),
        make_result("lean", recall=0.80, correctness=0.80, total_latency_ms=1500.0, cost_per_query=0.002),
    ])
    t = report.tradeoffs[0]
    assert t.summary == "faster and/or cheaper, but lower quality"


# --- format_delta ---------------------------------------------------------------------
def test_format_delta_units():
    assert "ms" in format_delta(_delta(2000.0, 2320.0, False), "ms")
    assert "$" in format_delta(_delta(0.001, 0.005, False), "$")
    assert "%" in format_delta(_delta(0.8, 0.88, True), "")
    assert "[better]" in format_delta(_delta(0.8, 0.88, True), "")
