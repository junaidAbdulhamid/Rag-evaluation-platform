"""Tests for the deterministic generation scorers."""

import pytest

from app.evaluation.generation_metrics import (
    exact_match,
    is_abstention,
    normalize_answer,
    number_coverage,
    token_f1,
    token_recall,
)
from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY


# --- normalize_answer ---------------------------------------------------------------
def test_normalize_lowercases_strips_punctuation_and_articles():
    assert normalize_answer("The  30-day, refund!") == "30 day refund"


# --- exact_match -------------------------------------------------------------------------
def test_exact_match_ignores_case_punctuation_articles():
    assert exact_match("30 days.", "The 30 days") == 1.0
    assert exact_match("30 days", "31 days") == 0.0


# --- token_f1 / token_recall ------------------------------------------------------------
def test_token_recall_measures_reference_coverage():
    # every reference token appears in the prediction
    assert token_recall("refunds take 5 to 7 business days", "5 to 7 business days") == 1.0
    # reference = [5, to, 7, business]; prediction shares [5, business] -> 2/4
    assert token_recall("5 business days", "5 to 7 business") == pytest.approx(0.5)


def test_token_f1_zero_when_no_overlap():
    assert token_f1("completely different text", "thirty calendar days") == 0.0


def test_token_f1_between_zero_and_one_for_partial_overlap():
    value = token_f1("the warranty lasts 12 months", "12 months from purchase")
    assert 0.0 < value < 1.0


def test_token_scores_zero_for_empty_prediction():
    assert token_recall("", "some reference") == 0.0
    assert token_f1("", "some reference") == 0.0


# --- number_coverage ---------------------------------------------------------------------
def test_number_coverage_all_numbers_present():
    assert number_coverage("It costs $39.99 and adds 24 months", "$39.99 for 24 months") == 1.0


def test_number_coverage_partial():
    assert number_coverage("about 24 months", "$39.99 for 24 months") == 0.5


def test_number_coverage_is_none_when_reference_has_no_numbers():
    assert number_coverage("ships to the US and Canada", "United States and Canada only") is None


def test_number_coverage_is_digit_boundary_aware():
    # reference number "2" must not be counted as present just because "2024" appears
    assert number_coverage("in the year 2024", "you get 2 free months") == 0.0


# --- is_abstention -----------------------------------------------------------------------
def test_is_abstention_matches_the_canonical_reply():
    assert is_abstention(INSUFFICIENT_CONTEXT_REPLY) is True
    assert is_abstention("Sorry, the provided context is insufficient here.") is True


def test_is_abstention_false_for_a_real_answer():
    assert is_abstention("Customers have 30 days to request a refund.") is False
