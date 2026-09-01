"""Tests for failure diagnosis + leaderboards."""

from __future__ import annotations

from typing import Optional

from app.evaluation.citation import CitationEvaluationResult, CitationLink, CitedClaim
from app.evaluation.citation_eval import QuestionCitationResult
from app.evaluation.faithfulness import Claim, FaithfulnessResult
from app.evaluation.faithfulness_eval import QuestionFaithfulnessResult
from app.evaluation.generation import DeterministicScores, QuestionGenerationResult
from app.evaluation.judge import GenerationJudgement
from app.evaluation.retrieval import QuestionRetrievalResult, RetrievalMetrics
from app.experiment.config import ExperimentConfig
from app.experiment.failure_analysis import (
    FailureCategory,
    FailureThresholds,
    analyze_failures,
    diagnose_question,
)
from app.experiment.results import ExperimentResult, QuestionCost, QuestionExperimentResult


def q(
    qid: str,
    *,
    hit_rate: float = 1.0,
    recall: Optional[float] = 1.0,
    relevant: Optional[list[str]] = None,
    retrieved_docs: Optional[list[str]] = None,
    correctness: Optional[float] = 1.0,
    abstained: bool = False,
    abstention_expected: bool = False,
    faithfulness: Optional[float] = 1.0,
    hallucinated_markers: int = 0,
    citation_precision: Optional[float] = None,
    latency_total: float = 1000.0,
    cost: float = 0.001,
    error: Optional[str] = None,
) -> QuestionExperimentResult:
    relevant = relevant if relevant is not None else (["doc_a"] if not abstention_expected else [])
    retrieved_docs = retrieved_docs if retrieved_docs is not None else (relevant or ["other"])

    retrieval = None
    if not error:
        retrieval = QuestionRetrievalResult(
            question_id=qid,
            retrieved_doc_ids=retrieved_docs,
            relevant_doc_ids=relevant,
            metrics=RetrievalMetrics(
                k=4, hit_rate=hit_rate, precision=0.5, recall=recall, reciprocal_rank=1.0,
                ndcg=1.0, num_relevant=len(relevant), num_retrieved_docs=len(retrieved_docs),
            ),
        )
    generation = None
    if not error:
        generation = QuestionGenerationResult(
            question_id=qid, question=f"question {qid}", expected_answer="a", generated_answer="b",
            deterministic=DeterministicScores(
                exact_match=1.0, token_f1=1.0, token_recall=1.0, number_coverage=None,
                abstention_expected=abstention_expected, abstained=abstained,
                abstention_correct=(abstained == abstention_expected),
            ),
            judgement=(
                GenerationJudgement(
                    correctness=correctness, relevance=1.0,
                    correctness_reasoning="r", relevance_reasoning="r",
                )
                if correctness is not None
                else None
            ),
        )
    faith = None
    if faithfulness is not None and not error:
        n = 20  # 20 claims -> any multiple of 0.05 is an exact score
        supported = int(round(faithfulness * n))
        faith = QuestionFaithfulnessResult(
            question_id=qid, answer="b",
            result=FaithfulnessResult.from_claims(
                [Claim(text=f"c{i}", supported=(i < supported), reason="r") for i in range(n)]
            ),
        )
    cite = None
    if (hallucinated_markers or citation_precision is not None) and not error:
        links = [
            CitationLink(claim_index=0, claim_text="c", marker=1, exists=True,
                         resolved_chunk_id="x::1", supports_claim=True)
        ]
        links += [
            CitationLink(claim_index=0, claim_text="c", marker=9, exists=False,
                         resolved_chunk_id=None, supports_claim=None)
            for _ in range(hallucinated_markers)
        ]
        cite = QuestionCitationResult(
            question_id=qid, answer="b",
            result=CitationEvaluationResult.compute(
                [CitedClaim(text="c", markers=[1], has_citation=True)], links
            ),
        )

    return QuestionExperimentResult(
        question_id=qid,
        question=f"question {qid}",
        retrieved_chunk_ids=[],
        retrieved_doc_ids=retrieved_docs,
        generated_answer="b",
        retrieval=retrieval,
        generation=generation,
        faithfulness=faith,
        citation=cite,
        latency_ms={"total": latency_total},
        cost=QuestionCost(total_usd=cost),
        estimated_cost_usd=cost,
        error=error,
    )


def experiment(questions) -> ExperimentResult:
    return ExperimentResult(
        experiment_id="exp",
        config=ExperimentConfig(experiment_name="exp"),
        started_at="2026-09-01T00:00:00+00:00",
        finished_at="2026-09-01T00:05:00+00:00",
        num_questions=len(questions),
        num_errors=sum(1 for x in questions if x.error),
        document_count=6,
        chunk_count=12,
        per_question=questions,
    )


THR = FailureThresholds()


# --- diagnose_question -------------------------------------------------------------------
def test_ok_question():
    d = diagnose_question(q("q1"), THR)
    assert d.category is FailureCategory.OK and not d.is_failure


def test_retrieval_failure_when_nothing_relevant_retrieved():
    d = diagnose_question(q("q1", hit_rate=0.0, recall=0.0, retrieved_docs=["wrong"]), THR)
    assert d.category is FailureCategory.RETRIEVAL_FAILURE and d.is_failure
    assert "not retrieved" in d.reason


def test_generation_failure_when_retrieved_but_wrong():
    d = diagnose_question(q("q1", hit_rate=1.0, correctness=0.2, faithfulness=1.0), THR)
    assert d.category is FailureCategory.GENERATION_FAILURE


def test_hallucination_from_low_faithfulness():
    d = diagnose_question(q("q1", correctness=0.9, faithfulness=0.25), THR)
    assert d.category is FailureCategory.HALLUCINATION
    assert "unsupported" in d.reason


def test_hallucination_when_answering_an_unanswerable_question():
    d = diagnose_question(q("q1", abstention_expected=True, abstained=False, faithfulness=None), THR)
    assert d.category is FailureCategory.HALLUCINATION


def test_insufficient_context_is_not_a_failure():
    d = diagnose_question(q("q1", abstention_expected=True, abstained=True, faithfulness=None), THR)
    assert d.category is FailureCategory.INSUFFICIENT_CONTEXT
    assert d.is_failure is False


def test_citation_failure_when_answer_plausible_but_citations_hallucinated():
    d = diagnose_question(
        q("q1", correctness=0.9, faithfulness=1.0, hallucinated_markers=2), THR
    )
    assert d.category is FailureCategory.CITATION_FAILURE


def test_error_question():
    d = diagnose_question(q("q1", error="RuntimeError('x')"), THR)
    assert d.category is FailureCategory.ERROR and d.is_failure


def test_priority_retrieval_failure_beats_generation_failure():
    # both retrieval missed AND correctness is low -> retrieval is the primary cause
    d = diagnose_question(q("q1", hit_rate=0.0, recall=0.0, correctness=0.1), THR)
    assert d.category is FailureCategory.RETRIEVAL_FAILURE


def test_thresholds_are_configurable():
    strict = FailureThresholds(faithfulness_min=0.99)
    d = diagnose_question(q("q1", correctness=0.9, faithfulness=0.9), strict)
    assert d.category is FailureCategory.HALLUCINATION  # 0.9 < 0.99 now


# --- analyze_failures -------------------------------------------------------------------
def test_analysis_counts_and_leaderboards():
    fa = analyze_failures(
        experiment([
            q("ok"),
            q("retr", hit_rate=0.0, recall=0.0, retrieved_docs=["wrong"]),
            q("gen", correctness=0.2),
            q("halluc", correctness=0.9, faithfulness=0.25),
            q("slow", latency_total=9000.0),
            q("pricey", cost=0.05),
        ]),
        top_n=2,
    )

    assert fa.category_counts["RETRIEVAL_FAILURE"] == 1
    assert fa.category_counts["GENERATION_FAILURE"] == 1
    assert fa.category_counts["HALLUCINATION"] == 1
    assert fa.retrieval_failures == ["retr"]
    assert fa.generation_failures == ["gen"]

    assert fa.lowest_recall[0].question_id == "retr" and fa.lowest_recall[0].value == 0.0
    assert fa.lowest_correctness[0].question_id == "gen"
    assert fa.highest_latency[0].question_id == "slow"
    assert fa.highest_cost[0].question_id == "pricey"
    assert len(fa.highest_latency) == 2  # top_n respected

    assert {d.question_id for d in fa.failures} == {"retr", "gen", "halluc"}
