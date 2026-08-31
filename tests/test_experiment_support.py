"""Tests for the experiment support pieces: token metering, timing, cost."""

import time

from app.experiment.cost import estimate_cost
from app.experiment.metering import RecordingTextLLM, TokenMeter, add_usage
from app.experiment.timing import record_ms
from app.models import TokenUsage
from tests.fakes import FakeTextLLM


# --- TokenMeter / RecordingTextLLM ------------------------------------------------
def test_recording_llm_accumulates_usage_into_meter():
    meter = TokenMeter()
    llm = RecordingTextLLM(FakeTextLLM(["a", "b"]), meter)

    llm.complete("p1")
    before = meter.snapshot()
    llm.complete("p2")

    assert meter.calls == 2
    assert meter.usage.total_tokens == 4  # FakeTextLLM reports total=2 per call
    assert meter.delta_since(before).total_tokens == 2


def test_add_usage_handles_none():
    base = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert add_usage(base, None) == base
    assert add_usage(base, base).total_tokens == 6


# --- record_ms -------------------------------------------------------------------------
def test_record_ms_writes_a_positive_duration():
    store: dict = {}
    with record_ms(store, "work"):
        time.sleep(0.005)
    assert store["work"] >= 4.0  # ~5ms, allow scheduling slack


def test_record_ms_records_even_on_exception():
    store: dict = {}
    try:
        with record_ms(store, "boom"):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    assert "boom" in store


# --- estimate_cost -------------------------------------------------------------------
def test_estimate_cost_known_model():
    # opus-5: $5/1M input, $25/1M output
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    assert estimate_cost(usage, "claude-opus-5") == 30.0


def test_estimate_cost_unknown_model_is_zero():
    assert estimate_cost(TokenUsage(prompt_tokens=999, completion_tokens=999), "mystery") == 0.0


def test_estimate_cost_zero_usage():
    assert estimate_cost(TokenUsage(), "claude-opus-5") == 0.0
