"""Tests for the citation evaluator and its aggregation."""

import json

import pytest

from app.evaluation.citation import (
    CitationEvaluationResult,
    CitationLink,
    CitedClaim,
    LLMCitationEvaluator,
)
from app.evaluation.citation_eval import aggregate_citation_metrics, evaluate_citations
from app.models import Chunk, RetrievedChunk
from tests.fakes import FakeCitationEvaluator, FakeTextLLM


def ctx(doc_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::c{rank}", document_id=doc_id, text=f"text {rank}"),
        score=1.0,
        rank=rank,
    )


RETRIEVED = [ctx("refund_policy", 1), ctx("payments", 2)]


def assessment(text, markers, supported):
    return {"text": text, "markers": markers, "supported_markers": supported}


def llm_reply(*assessments):
    return FakeTextLLM(json.dumps({"claims": list(assessments)}))


# --- LLMCitationEvaluator ------------------------------------------------------------
def test_one_good_citation_one_missing():
    # answer must actually contain the markers the model claims
    answer = "Refunds take 30 days. [1] Refunds need a receipt."
    llm = llm_reply(
        assessment("Refunds take 30 days.", [1], [1]),
        assessment("Refunds need a receipt.", [], []),
    )
    res = LLMCitationEvaluator(llm).evaluate(answer=answer, retrieved=RETRIEVED)

    assert res.num_claims == 2
    assert res.num_claims_with_citation == 1
    assert res.citation_completeness == 0.5
    assert res.citation_precision == 1.0        # 1 link, supported
    assert res.citation_correctness == 1.0      # the one cited claim is backed
    assert res.citation_hallucination_rate == 0.0


def test_hallucinated_marker_counts_against_precision_and_halluc_rate():
    answer = "Big claim. [9]"
    llm = llm_reply(assessment("Big claim.", [9], []))
    res = LLMCitationEvaluator(llm).evaluate(answer=answer, retrieved=RETRIEVED)

    assert res.num_citation_links == 1
    assert res.num_hallucinated_links == 1
    assert res.citation_hallucination_rate == 1.0
    assert res.citation_precision == 0.0
    assert res.citation_correctness == 0.0
    assert res.links[0].exists is False
    assert res.links[0].supports_claim is None


def test_marker_present_but_source_does_not_support_claim():
    answer = "Wrong thing. [2]"
    llm = llm_reply(assessment("Wrong thing.", [2], []))  # cited but not supported
    res = LLMCitationEvaluator(llm).evaluate(answer=answer, retrieved=RETRIEVED)

    assert res.citation_completeness == 1.0
    assert res.citation_precision == 0.0
    assert res.citation_correctness == 0.0
    assert res.num_hallucinated_links == 0


def test_marker_claimed_by_model_but_absent_from_text_is_dropped():
    answer = "A claim with no bracket."
    llm = llm_reply(assessment("A claim with no bracket.", [1], [1]))  # [1] not in the text
    res = LLMCitationEvaluator(llm).evaluate(answer=answer, retrieved=RETRIEVED)

    assert res.claims[0].markers == []
    assert res.num_citation_links == 0
    assert res.citation_completeness == 0.0


def test_no_factual_claims_gives_all_none():
    llm = llm_reply()  # empty claims list
    res = LLMCitationEvaluator(llm).evaluate(answer="I cannot answer that.", retrieved=RETRIEVED)

    assert res.num_claims == 0
    assert res.citation_completeness is None
    assert res.citation_precision is None
    assert res.citation_correctness is None
    assert res.citation_hallucination_rate is None


# --- CitationEvaluationResult.compute (direct) ----------------------------------------
def test_compute_multi_link_claim_correct_if_any_link_supported():
    claims = [CitedClaim(text="c", markers=[1, 2], has_citation=True)]
    links = [
        CitationLink(claim_index=0, claim_text="c", marker=1, exists=True,
                     resolved_chunk_id="x::1", supports_claim=False),
        CitationLink(claim_index=0, claim_text="c", marker=2, exists=True,
                     resolved_chunk_id="x::2", supports_claim=True),
    ]
    res = CitationEvaluationResult.compute(claims, links)

    assert res.citation_precision == 0.5     # 1 of 2 links supported
    assert res.citation_correctness == 1.0   # claim has >=1 supported link


# --- aggregate + driver -------------------------------------------------------------------
def test_aggregate_macro_and_micro():
    good = CitationEvaluationResult.compute(
        [CitedClaim(text="a", markers=[1], has_citation=True)],
        [CitationLink(claim_index=0, claim_text="a", marker=1, exists=True,
                      resolved_chunk_id="x::1", supports_claim=True)],
    )
    halluc = CitationEvaluationResult.compute(
        [CitedClaim(text="b", markers=[9], has_citation=True)],
        [CitationLink(claim_index=0, claim_text="b", marker=9, exists=False,
                      resolved_chunk_id=None, supports_claim=None)],
    )
    empty = CitationEvaluationResult.compute([], [])

    evaluator = FakeCitationEvaluator({"a1": good, "a2": halluc, "a3": empty})
    cases = [("q1", "a1", []), ("q2", "a2", []), ("q3", "a3", [])]

    agg = evaluate_citations(cases, evaluator).aggregate

    assert agg.num_questions == 3
    assert agg.num_scored == 2                       # q3 has no claims
    assert agg.citation_precision == pytest.approx(0.5)   # macro: (1.0 + 0.0) / 2
    assert agg.total_links == 2
    assert agg.total_hallucinated_links == 1


def test_aggregate_empty_raises():
    with pytest.raises(ValueError):
        aggregate_citation_metrics([])
