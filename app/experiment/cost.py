"""Minimal cost estimation.

Phase 11 replaces this with a proper, configurable pricing registry (embedding vs
prompt vs completion vs evaluation cost). For now: a small table of USD-per-million
-token rates and one function, so `run_experiment` can put a dollar figure on a run.

Rates below are Anthropic first-party list prices (input / output per 1M tokens).
"""

from __future__ import annotations

from app.models import TokenUsage

# {model_id: (input_per_million_usd, output_per_million_usd)}
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


def estimate_cost(usage: TokenUsage, model: str) -> float:
    """USD cost of `usage` at `model`'s rate. Unknown model -> 0.0 (not priced)."""
    rate = MODEL_PRICING.get(model)
    if rate is None:
        return 0.0
    input_rate, output_rate = rate
    return (
        usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate
    ) / 1_000_000
