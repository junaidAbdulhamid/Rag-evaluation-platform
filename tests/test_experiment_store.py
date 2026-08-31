"""Tests for the SQLite experiment store: save / list / get / delete round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.judge import GenerationJudgement
from app.experiment.config import ExperimentConfig
from app.experiment.results import (
    ExperimentError,
    ExperimentResult,
    LatencySummary,
    QuestionExperimentResult,
)
from app.experiment.store import ExperimentStore
from app.evaluation.generation import (
    AggregateGenerationMetrics,
    DeterministicScores,
    QuestionGenerationResult,
)
from app.evaluation.retrieval import (
    AggregateRetrievalMetrics,
    QuestionRetrievalResult,
    RetrievalMetrics,
)
from app.models import TokenUsage


def make_result(experiment_id: str = "exp_1", *, recall: float = 0.9) -> ExperimentResult:
    q = QuestionExperimentResult(
        question_id="q1",
        question="How long for a refund?",
        retrieved_chunk_ids=["refund_policy::chunk_0", "refund_policy::chunk_1"],
        retrieved_doc_ids=["refund_policy"],
        generated_answer="Customers have 30 days.",
        retrieval=QuestionRetrievalResult(
            question_id="q1",
            retrieved_doc_ids=["refund_policy"],
            relevant_doc_ids=["refund_policy"],
            metrics=RetrievalMetrics(
                k=4, hit_rate=1.0, precision=0.5, recall=recall, reciprocal_rank=1.0,
                ndcg=1.0, num_relevant=1, num_retrieved_docs=1,
            ),
        ),
        generation=QuestionGenerationResult(
            question_id="q1", question="q?", expected_answer="a", generated_answer="b",
            deterministic=DeterministicScores(
                exact_match=1.0, token_f1=1.0, token_recall=1.0, number_coverage=1.0,
                abstention_expected=False, abstained=False, abstention_correct=True,
            ),
            judgement=GenerationJudgement(
                correctness=0.9, relevance=1.0,
                correctness_reasoning="ok", relevance_reasoning="ok",
            ),
        ),
        latency_ms={"retrieval": 5.0, "generation": 800.0, "evaluation": 900.0, "total": 1705.0},
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        estimated_cost_usd=0.001,
    )
    return ExperimentResult(
        experiment_id=experiment_id,
        config=ExperimentConfig(experiment_name="base", chunk_size=500, top_k=4),
        started_at="2026-08-30T10:00:00+00:00",
        finished_at="2026-08-30T10:05:00+00:00",
        num_questions=1,
        num_errors=0,
        document_count=6,
        chunk_count=12,
        per_question=[q],
        retrieval=AggregateRetrievalMetrics(
            k=4, num_questions_total=1, num_questions_scored=1, hit_rate=1.0, precision=0.5,
            recall=recall, mrr=1.0, ndcg=1.0,
        ),
        generation=AggregateGenerationMetrics(
            num_questions=1, exact_match=1.0, token_f1=1.0, token_recall=1.0, number_coverage=1.0,
            abstention_accuracy=1.0, num_with_numbers=1, num_judged=1,
            judge_correctness=0.9, judge_relevance=1.0,
        ),
        latency=LatencySummary(retrieval_ms=5.0, generation_ms=800.0, evaluation_ms=900.0, total_ms=1705.0),
        total_token_usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        estimated_cost_usd=0.001,
        errors=[],
    )


@pytest.fixture
def store(tmp_path: Path) -> ExperimentStore:
    s = ExperimentStore(tmp_path / "experiments.db")
    yield s
    s.close()


def test_save_then_get_round_trips_full_object(store: ExperimentStore):
    original = make_result("exp_round")
    store.save(original)

    loaded = store.get("exp_round")
    assert loaded == original  # exact Pydantic equality


def test_get_missing_returns_none(store: ExperimentStore):
    assert store.get("nope") is None


def test_list_projects_headline_columns_newest_first(store: ExperimentStore):
    store.save(make_result("exp_old", recall=0.5))
    r2 = make_result("exp_new", recall=0.95)
    r2.started_at = "2026-08-31T00:00:00+00:00"
    store.save(r2)

    rows = store.list()
    assert [r.experiment_id for r in rows] == ["exp_new", "exp_old"]
    assert rows[0].retrieval_recall == 0.95
    assert rows[0].judge_correctness == 0.9
    assert rows[0].generation_model == "claude-opus-5"


def test_save_is_idempotent_on_experiment_id(store: ExperimentStore):
    store.save(make_result("exp_x", recall=0.4))
    store.save(make_result("exp_x", recall=0.8))  # re-save, same id

    assert store.count() == 1
    assert store.get("exp_x").per_question[0].retrieval.metrics.recall == 0.8
    # question rows were replaced, not duplicated
    n = store._conn.execute(
        "SELECT COUNT(*) FROM question_results WHERE experiment_id = 'exp_x'"
    ).fetchone()[0]
    assert n == 1


def test_question_rows_store_queryable_columns(store: ExperimentStore):
    store.save(make_result("exp_q"))
    row = store._conn.execute(
        "SELECT * FROM question_results WHERE experiment_id = 'exp_q'"
    ).fetchone()

    assert row["question_id"] == "q1"
    assert json.loads(row["retrieved_chunk_ids"]) == ["refund_policy::chunk_0", "refund_policy::chunk_1"]
    assert row["retrieval_recall"] == 0.9
    assert row["judge_correctness"] == 0.9
    assert row["latency_total_ms"] == 1705.0


def test_delete_cascades_to_question_rows(store: ExperimentStore):
    store.save(make_result("exp_del"))
    assert store.delete("exp_del") is True
    assert store.get("exp_del") is None
    assert store._conn.execute(
        "SELECT COUNT(*) FROM question_results WHERE experiment_id = 'exp_del'"
    ).fetchone()[0] == 0
    assert store.delete("exp_del") is False  # already gone


def test_errored_question_persists_with_null_metrics(store: ExperimentStore):
    result = make_result("exp_err")
    result.per_question.append(
        QuestionExperimentResult(question_id="q2", question="boom", error="RuntimeError('x')")
    )
    result.num_errors = 1
    result.errors = [ExperimentError(question_id="q2", stage="run", message="RuntimeError('x')")]
    store.save(result)

    loaded = store.get("exp_err")
    assert loaded.num_errors == 1
    assert loaded.per_question[1].error is not None
    row = store._conn.execute(
        "SELECT * FROM question_results WHERE experiment_id='exp_err' AND question_id='q2'"
    ).fetchone()
    assert row["error"] is not None
    assert row["retrieval_recall"] is None
