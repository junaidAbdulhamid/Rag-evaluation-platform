"""Tests for the pricing registry and cost functions."""

from __future__ import annotations

import json

import pytest

from app.experiment import cost as pricing
from app.experiment.cost import (
    ModelPrice,
    apply_configured_pricing,
    embedding_cost,
    embedding_rate_per_million,
    llm_cost,
    model_price,
)
from app.models import TokenUsage


def test_llm_cost_known_model():
    # opus-5: $5/1M input, $25/1M output
    usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert llm_cost(usage, "claude-opus-5") == 30.0


def test_llm_cost_unknown_model_is_zero():
    assert llm_cost(TokenUsage(prompt_tokens=999, completion_tokens=999), "mystery") == 0.0


def test_llm_cost_ignores_embedding_tokens():
    usage = TokenUsage(embedding_tokens=1_000_000, prompt_tokens=0, completion_tokens=0)
    assert llm_cost(usage, "claude-opus-5") == 0.0


def test_embedding_cost_local_model_is_free():
    assert embedding_cost(5_000_000, "sentence-transformers/all-MiniLM-L6-v2") == 0.0


def test_embedding_cost_priced_model():
    # text-embedding-3-small: $0.02 / 1M
    assert embedding_cost(1_000_000, "text-embedding-3-small") == pytest.approx(0.02)


def test_unknown_embedding_model_assumed_free():
    assert embedding_rate_per_million("some-local-thing") == 0.0


def test_model_price_lookup():
    assert model_price("claude-haiku-4-5") == ModelPrice(input_per_million=1.0, output_per_million=5.0)
    assert model_price("nope") is None


def test_configured_pricing_overrides_merge(tmp_path, monkeypatch):
    overrides = tmp_path / "prices.json"
    overrides.write_text(json.dumps({
        "models": {"my-model": {"input_per_million": 7.0, "output_per_million": 21.0}},
        "embeddings": {"my-embed": 0.5},
    }))
    monkeypatch.setattr(pricing.settings, "pricing_file", str(overrides))
    try:
        apply_configured_pricing()
        assert llm_cost(TokenUsage(prompt_tokens=1_000_000), "my-model") == 7.0
        assert embedding_rate_per_million("my-embed") == 0.5
    finally:
        pricing.MODEL_PRICING.pop("my-model", None)
        pricing.EMBEDDING_PRICING.pop("my-embed", None)
