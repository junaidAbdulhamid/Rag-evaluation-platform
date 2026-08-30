"""Tests for the shared structured-output helper (JSON extraction + retry loop)."""

import json

import pytest
from pydantic import BaseModel, Field

from app.evaluation.structured_output import (
    StructuredOutputError,
    extract_json_object,
    retry_structured_call,
)
from tests.fakes import FakeTextLLM


class Score(BaseModel):
    value: float = Field(ge=0.0, le=1.0)


# --- extract_json_object -------------------------------------------------------------
def test_extract_plain_object():
    assert extract_json_object('{"value": 0.5}') == '{"value": 0.5}'


def test_extract_from_code_fence_and_prose():
    text = 'Sure!\n```json\n{"value": 0.5}\n```\ndone'
    assert json.loads(extract_json_object(text)) == {"value": 0.5}


def test_extract_raises_when_no_object():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("no braces here")


# --- retry_structured_call --------------------------------------------------------------
def test_succeeds_first_try():
    llm = FakeTextLLM('{"value": 0.4}')
    out = retry_structured_call(llm, prompt="p", response_model=Score)
    assert out.value == 0.4
    assert len(llm.calls) == 1


def test_retries_then_succeeds_and_nudges():
    llm = FakeTextLLM(["garbage", '{"value": 9}', '{"value": 0.4}'])  # bad, invalid, good
    out = retry_structured_call(llm, prompt="p", response_model=Score, max_retries=2)
    assert out.value == 0.4
    assert len(llm.calls) == 3
    assert "could not be parsed" in llm.calls[1]


def test_raises_after_exhausting_retries():
    llm = FakeTextLLM("never valid")
    with pytest.raises(StructuredOutputError):
        retry_structured_call(llm, prompt="p", response_model=Score, max_retries=2)
    assert len(llm.calls) == 3


def test_custom_error_class():
    class MyError(StructuredOutputError):
        pass

    with pytest.raises(MyError):
        retry_structured_call(
            FakeTextLLM("bad"), prompt="p", response_model=Score, max_retries=0, error_cls=MyError
        )
