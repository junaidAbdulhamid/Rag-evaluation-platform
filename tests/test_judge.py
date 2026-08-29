"""Tests for the LLM-as-judge: structured parsing, validation, and the retry loop.
All run against a FakeTextLLM - no network."""

import json

import pytest

from app.evaluation.judge import (
    GenerationJudgement,
    JudgeParseError,
    LLMGenerationJudge,
    parse_judgement,
)
from tests.fakes import FakeTextLLM

VALID_JSON = json.dumps(
    {
        "correctness": 0.8,
        "relevance": 1.0,
        "correctness_reasoning": "Matches the reference on the key figure.",
        "relevance_reasoning": "Directly answers the question asked.",
    }
)


def run_judge(responses, max_retries=2):
    llm = FakeTextLLM(responses)
    judge = LLMGenerationJudge(llm, max_retries=max_retries)
    result = judge.judge(
        question="How many days for a refund?",
        expected_answer="30 days.",
        generated_answer="You have 30 days.",
    )
    return result, llm


# --- parse_judgement -------------------------------------------------------------------
def test_parses_plain_json():
    j = parse_judgement(VALID_JSON)
    assert (j.correctness, j.relevance) == (0.8, 1.0)


def test_parses_json_wrapped_in_code_fences_and_prose():
    text = f"Here is my assessment:\n```json\n{VALID_JSON}\n```\nThanks!"
    assert parse_judgement(text).correctness == 0.8


def test_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        GenerationJudgement.model_validate({**json.loads(VALID_JSON), "correctness": 1.5})


def test_rejects_unknown_key():
    with pytest.raises(ValueError):
        GenerationJudgement.model_validate({**json.loads(VALID_JSON), "confidence": 0.5})


def test_rejects_blank_reasoning():
    with pytest.raises(ValueError):
        GenerationJudgement.model_validate({**json.loads(VALID_JSON), "relevance_reasoning": "  "})


def test_no_json_object_raises_decode_error():
    with pytest.raises(json.JSONDecodeError):
        parse_judgement("I think it's pretty good, maybe 8 out of 10.")


# --- LLMGenerationJudge retry loop -------------------------------------------------------
def test_succeeds_on_first_try():
    result, llm = run_judge(VALID_JSON)
    assert result.correctness == 0.8
    assert len(llm.calls) == 1


def test_retries_then_succeeds():
    result, llm = run_judge(["not json at all", "still broken", VALID_JSON])
    assert result.relevance == 1.0
    assert len(llm.calls) == 3
    # the retry prompt tells the model its previous output was unparseable
    assert "could not be parsed" in llm.calls[1]


def test_raises_after_exhausting_retries():
    with pytest.raises(JudgeParseError):
        run_judge(["nope", "nope", "nope", "nope"], max_retries=2)  # 1 + 2 attempts


def test_invalid_structure_also_triggers_retry():
    bad = json.dumps({"correctness": 5, "relevance": 0.5})  # out of range + missing fields
    result, llm = run_judge([bad, VALID_JSON])
    assert len(llm.calls) == 2
    assert result.correctness == 0.8
