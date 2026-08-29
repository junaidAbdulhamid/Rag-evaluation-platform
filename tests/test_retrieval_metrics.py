"""Edge-case tests for the from-scratch retrieval metric functions.

Covers every case the Phase 3 spec calls out: nothing relevant retrieved, all
relevant, relevant at rank 1, relevant at a lower rank, and k larger than the number
of retrieved results.
"""

import math

import pytest

from app.evaluation.retrieval_metrics import (
    dcg_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RELEVANT = {"docA", "docB"}


# --- hit_rate_at_k -------------------------------------------------------------------
def test_hit_rate_relevant_in_top_k():
    assert hit_rate_at_k(["docX", "docA", "docY"], RELEVANT, k=2) == 1.0


def test_hit_rate_relevant_below_cutoff():
    assert hit_rate_at_k(["docX", "docY", "docA"], RELEVANT, k=2) == 0.0
    assert hit_rate_at_k(["docX", "docY", "docA"], RELEVANT, k=3) == 1.0


def test_hit_rate_none_relevant_retrieved():
    assert hit_rate_at_k(["docX", "docY"], RELEVANT, k=5) == 0.0


def test_hit_rate_rejects_bad_k():
    with pytest.raises(ValueError):
        hit_rate_at_k(["docA"], RELEVANT, k=0)


# --- precision_at_k ----------------------------------------------------------------------
def test_precision_all_retrieved_relevant():
    assert precision_at_k(["docA", "docB"], RELEVANT, k=2) == 1.0


def test_precision_half_relevant():
    assert precision_at_k(["docA", "docX"], RELEVANT, k=2) == 0.5


def test_precision_none_relevant():
    assert precision_at_k(["docX", "docY"], RELEVANT, k=2) == 0.0


def test_precision_k_larger_than_results_divides_by_actual_count():
    # only 1 doc retrieved, k=5 -> denominator is 1, not 5
    assert precision_at_k(["docA"], RELEVANT, k=5) == 1.0


def test_precision_empty_retrieval_is_zero():
    assert precision_at_k([], RELEVANT, k=3) == 0.0


# --- recall_at_k -----------------------------------------------------------------------
def test_recall_finds_all_relevant():
    assert recall_at_k(["docA", "docB", "docX"], RELEVANT, k=3) == 1.0


def test_recall_finds_some_relevant():
    assert recall_at_k(["docA", "docX"], RELEVANT, k=2) == 0.5


def test_recall_finds_none():
    assert recall_at_k(["docX", "docY"], RELEVANT, k=2) == 0.0


def test_recall_k_larger_than_results():
    assert recall_at_k(["docA"], RELEVANT, k=10) == 0.5  # 1 of 2 relevant


def test_recall_is_none_when_no_relevant_documents_exist():
    assert recall_at_k(["docX", "docY"], set(), k=3) is None


# --- reciprocal_rank -------------------------------------------------------------------
def test_reciprocal_rank_relevant_at_rank_1():
    assert reciprocal_rank(["docA", "docX", "docY"], RELEVANT) == 1.0


def test_reciprocal_rank_relevant_at_lower_ranks():
    assert reciprocal_rank(["docX", "docA"], RELEVANT) == 0.5
    assert reciprocal_rank(["docX", "docY", "docA"], RELEVANT) == pytest.approx(1 / 3)


def test_reciprocal_rank_uses_first_relevant_hit():
    assert reciprocal_rank(["docX", "docA", "docB"], RELEVANT) == 0.5


def test_reciprocal_rank_no_relevant_retrieved():
    assert reciprocal_rank(["docX", "docY"], RELEVANT) == 0.0


# --- dcg_at_k --------------------------------------------------------------------------
def test_dcg_rank_one_is_undiscounted():
    assert dcg_at_k([1.0, 0.0, 0.0], k=3) == 1.0


def test_dcg_later_positions_are_discounted():
    assert dcg_at_k([0.0, 0.0, 1.0], k=3) == pytest.approx(1 / math.log2(4))  # == 0.5


def test_dcg_sums_positions():
    expected = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert dcg_at_k([1.0, 1.0, 1.0], k=3) == pytest.approx(expected)


def test_dcg_rejects_bad_k():
    with pytest.raises(ValueError):
        dcg_at_k([1.0], k=0)


# --- ndcg_at_k -------------------------------------------------------------------------
def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["docA", "docB", "docX"], RELEVANT, k=3) == pytest.approx(1.0)


def test_ndcg_all_documents_relevant_is_one():
    assert ndcg_at_k(["docA", "docB"], RELEVANT, k=2) == pytest.approx(1.0)


def test_ndcg_no_relevant_retrieved_is_zero():
    assert ndcg_at_k(["docX", "docY"], RELEVANT, k=2) == 0.0


def test_ndcg_rewards_earlier_placement():
    early = ndcg_at_k(["docA", "docX", "docY"], {"docA"}, k=3)
    late = ndcg_at_k(["docX", "docY", "docA"], {"docA"}, k=3)
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(0.5)
    assert early > late


def test_ndcg_never_exceeds_one():
    for ranking in (["docA", "docB"], ["docB", "docA", "docX"], ["docX", "docA", "docB"]):
        value = ndcg_at_k(ranking, RELEVANT, k=3)
        assert 0.0 <= value <= 1.0


def test_ndcg_is_none_when_no_relevant_documents_exist():
    assert ndcg_at_k(["docX"], set(), k=3) is None


def test_ndcg_rejects_bad_k():
    with pytest.raises(ValueError):
        ndcg_at_k(["docA"], RELEVANT, k=-1)
