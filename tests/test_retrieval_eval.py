"""Tests for the retrieval-evaluation glue: chunk->doc reduction, per-question
scoring, aggregation, and the dataset-level driver."""

import pytest

from app.evaluation.dataset import EvalExample
from app.evaluation.retrieval import (
    aggregate_retrieval_metrics,
    doc_ranking_from_chunks,
    evaluate_retrieval,
    evaluate_retrieval_for_question,
)
from app.models import Chunk, RetrievedChunk


def rc(doc_id: str, rank: int, chunk_no: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::chunk_{chunk_no}", document_id=doc_id, text="..."),
        score=1.0 / rank,
        rank=rank,
    )


def example(qid: str, relevant: list[str]) -> EvalExample:
    return EvalExample(
        id=qid, question=f"question {qid}", expected_answer="a", relevant_document_ids=relevant
    )


# --- doc_ranking_from_chunks -------------------------------------------------------------
def test_doc_ranking_dedupes_keeping_best_rank():
    chunks = [rc("B", 1), rc("A", 2), rc("A", 3, chunk_no=1), rc("C", 4)]
    assert doc_ranking_from_chunks(chunks) == ["B", "A", "C"]


def test_doc_ranking_sorts_by_rank_even_if_input_unordered():
    chunks = [rc("C", 3), rc("A", 1), rc("B", 2)]
    assert doc_ranking_from_chunks(chunks) == ["A", "B", "C"]


def test_doc_ranking_empty():
    assert doc_ranking_from_chunks([]) == []


# --- evaluate_retrieval_for_question ------------------------------------------------------
def test_per_question_scoring_of_a_known_case():
    # retrieved docs in rank order: X, refund_policy, X  -> ranking [X, refund_policy]
    chunks = [rc("X", 1), rc("refund_policy", 2), rc("X", 3, chunk_no=1)]
    result = evaluate_retrieval_for_question(chunks, example("q1", ["refund_policy"]), k=3)

    assert result.retrieved_doc_ids == ["X", "refund_policy"]
    assert result.metrics.hit_rate == 1.0
    assert result.metrics.precision == 0.5          # 1 of 2 retrieved docs relevant
    assert result.metrics.recall == 1.0             # found the 1 relevant doc
    assert result.metrics.reciprocal_rank == 0.5    # relevant doc at effective rank 2
    assert 0.0 < result.metrics.ndcg < 1.0
    assert result.metrics.num_relevant == 1
    assert result.metrics.num_retrieved_docs == 2


def test_per_question_multiple_chunks_of_one_doc_do_not_inflate_recall():
    chunks = [rc("refund_policy", 1), rc("refund_policy", 2, chunk_no=1), rc("refund_policy", 3, chunk_no=2)]
    result = evaluate_retrieval_for_question(
        chunks, example("q1", ["refund_policy", "shipping_policy"]), k=3
    )
    assert result.retrieved_doc_ids == ["refund_policy"]
    assert result.metrics.recall == 0.5  # 1 of 2 relevant docs, NOT 3/2


def test_per_question_unanswerable_has_none_recall_and_ndcg():
    chunks = [rc("X", 1), rc("Y", 2)]
    result = evaluate_retrieval_for_question(chunks, example("q_un", []), k=2)

    assert result.is_unanswerable is True
    assert result.metrics.recall is None
    assert result.metrics.ndcg is None
    assert result.metrics.hit_rate == 0.0
    assert result.metrics.precision == 0.0
    assert result.metrics.reciprocal_rank == 0.0


# --- aggregate_retrieval_metrics ------------------------------------------------------
def test_aggregate_excludes_unanswerable_questions():
    good = evaluate_retrieval_for_question([rc("A", 1)], example("q1", ["A"]), k=1)
    miss = evaluate_retrieval_for_question([rc("X", 1)], example("q2", ["A"]), k=1)
    unans = evaluate_retrieval_for_question([rc("X", 1)], example("q3", []), k=1)

    agg = aggregate_retrieval_metrics([good, miss, unans])

    assert agg.num_questions_total == 3
    assert agg.num_questions_scored == 2
    assert agg.hit_rate == 0.5   # good=1, miss=0
    assert agg.recall == 0.5
    assert agg.mrr == 0.5
    assert agg.k == 1


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_retrieval_metrics([])


# --- evaluate_retrieval (dataset driver) --------------------------------------------
def test_evaluate_retrieval_end_to_end_with_a_fake_retriever():
    canned = {
        "question q1": [rc("A", 1), rc("B", 2)],
        "question q2": [rc("X", 1), rc("A", 2)],
        "question q3": [rc("X", 1)],  # unanswerable
    }

    def retrieve_fn(question: str, k: int):
        return canned[question][:k]

    dataset = [example("q1", ["A"]), example("q2", ["A"]), example("q3", [])]

    evaluation = evaluate_retrieval(dataset, retrieve_fn, k=2)

    assert [r.question_id for r in evaluation.per_question] == ["q1", "q2", "q3"]
    assert evaluation.aggregate.num_questions_scored == 2
    assert evaluation.aggregate.hit_rate == 1.0        # q1 & q2 both hit
    assert evaluation.aggregate.mrr == pytest.approx(0.75)  # q1: 1.0, q2: 0.5
    assert evaluation.k == 2
