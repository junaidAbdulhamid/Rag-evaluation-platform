"""Tests for the experiment support pieces: token metering, timing."""

import time

from app.experiment.metering import RecordingTextLLM, TokenMeter, add_usage
from app.models import TokenUsage
from app.observability.timing import record_ms
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


def test_add_usage_sums_all_fields_and_handles_none():
    base = TokenUsage(embedding_tokens=5, prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert add_usage(base, None) == base
    doubled = add_usage(base, base)
    assert (doubled.embedding_tokens, doubled.total_tokens) == (10, 6)


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
