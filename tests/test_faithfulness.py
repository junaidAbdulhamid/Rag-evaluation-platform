"""Tests for the LLM faithfulness evaluator - claim extraction, verification,
chunk-id linkage, and the no-claims / dropped-verdict edge cases. FakeTextLLM only."""

import json

import pytest

from app.evaluation.faithfulness import (
    LLMFaithfulnessEvaluator,
    blocks_to_chunk_ids,
)
from app.models import Chunk, RetrievedChunk
from tests.fakes import FakeTextLLM


def ctx(doc_id: str, text: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::c{rank}", document_id=doc_id, text=text),
        score=1.0,
        rank=rank,
    )


RETRIEVED = [
    ctx("refund_policy", "Refunds are available for 30 days.", 1),
    ctx("payment_methods", "We accept Visa and Mastercard.", 2),
]


def extract(claims: list[str]) -> str:
    return json.dumps({"claims": claims})


def verify(*verdicts: dict) -> str:
    return json.dumps({"verdicts": list(verdicts)})


# --- blocks_to_chunk_ids -----------------------------------------------------------------
def test_blocks_map_to_chunk_ids_and_drop_out_of_range():
    assert blocks_to_chunk_ids([1, 2, 5, 0], RETRIEVED) == [
        "refund_policy::c1",
        "payment_methods::c2",
    ]


# --- the two-step evaluate() ------------------------------------------------------------
def test_mixed_supported_and_unsupported_claims():
    llm = FakeTextLLM(
        [
            extract(["Refunds are available for 30 days.", "The original receipt is required."]),
            verify(
                {"claim_index": 0, "supported": True, "supporting_blocks": [1], "reason": "stated in block 1"},
                {"claim_index": 1, "supported": False, "supporting_blocks": [], "reason": "context is silent"},
            ),
        ]
    )
    result = LLMFaithfulnessEvaluator(llm).evaluate(answer="...", retrieved=RETRIEVED)

    assert result.num_claims == 2
    assert result.num_supported == 1
    assert result.score == 0.5
    assert result.claims[0].supported is True
    assert result.claims[0].supporting_chunk_ids == ["refund_policy::c1"]
    assert result.claims[1].supported is False
    assert result.claims[1].supporting_chunk_ids == []  # unsupported -> no citations kept


def test_no_factual_claims_yields_none_score_and_skips_verification():
    llm = FakeTextLLM([extract([])])  # extraction returns empty; verify never called
    result = LLMFaithfulnessEvaluator(llm).evaluate(answer="I cannot answer that.", retrieved=RETRIEVED)

    assert result.num_claims == 0
    assert result.score is None
    assert len(llm.calls) == 1  # no verification call


def test_verdicts_returned_out_of_order_are_matched_by_index():
    llm = FakeTextLLM(
        [
            extract(["claim A", "claim B"]),
            verify(
                {"claim_index": 1, "supported": False, "supporting_blocks": [], "reason": "b"},
                {"claim_index": 0, "supported": True, "supporting_blocks": [2], "reason": "a"},
            ),
        ]
    )
    result = LLMFaithfulnessEvaluator(llm).evaluate(answer="...", retrieved=RETRIEVED)

    assert result.claims[0].text == "claim A"
    assert result.claims[0].supported is True
    assert result.claims[0].supporting_chunk_ids == ["payment_methods::c2"]
    assert result.claims[1].text == "claim B"
    assert result.claims[1].supported is False


def test_missing_verdict_defaults_to_unsupported():
    llm = FakeTextLLM(
        [
            extract(["claim A", "claim B"]),
            verify({"claim_index": 0, "supported": True, "supporting_blocks": [1], "reason": "ok"}),
        ]
    )
    result = LLMFaithfulnessEvaluator(llm).evaluate(answer="...", retrieved=RETRIEVED)

    assert result.num_claims == 2
    assert result.claims[1].supported is False
    assert "no verdict" in result.claims[1].reason
    assert result.score == 0.5


def test_extraction_retries_on_bad_json():
    llm = FakeTextLLM(
        [
            "not json",
            extract(["Refunds are available for 30 days."]),
            verify({"claim_index": 0, "supported": True, "supporting_blocks": [1], "reason": "ok"}),
        ]
    )
    result = LLMFaithfulnessEvaluator(llm, max_retries=2).evaluate(answer="...", retrieved=RETRIEVED)

    assert result.score == 1.0
    assert len(llm.calls) == 3  # 1 failed + 1 good extraction, then 1 verification
