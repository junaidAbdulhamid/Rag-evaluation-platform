"""The pricing registry - the single place prices live.

Nothing else in the codebase should contain a dollar figure. Two registries:

* ``MODEL_PRICING``     - LLM input / output rates, USD per 1M tokens
* ``EMBEDDING_PRICING`` - embedding rate, USD per 1M tokens (local models are $0)

Rates below are Anthropic first-party list prices. "Configurable": call
``apply_configured_pricing()`` (the runner does) to merge a JSON overrides file
pointed at by ``settings.pricing_file``, so you can price a model the table doesn't
know, or use your negotiated rates, without editing code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from app.config import settings
from app.models import TokenUsage


class ModelPrice(BaseModel):
    input_per_million: float
    output_per_million: float


MODEL_PRICING: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(input_per_million=5.0, output_per_million=25.0),
    "claude-opus-4-8": ModelPrice(input_per_million=5.0, output_per_million=25.0),
    "claude-opus-4-7": ModelPrice(input_per_million=5.0, output_per_million=25.0),
    "claude-sonnet-5": ModelPrice(input_per_million=2.0, output_per_million=10.0),
    "claude-sonnet-4-6": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "claude-haiku-4-5": ModelPrice(input_per_million=1.0, output_per_million=5.0),
    "claude-fable-5": ModelPrice(input_per_million=10.0, output_per_million=50.0),
}

EMBEDDING_PRICING: dict[str, float] = {
    "sentence-transformers/all-MiniLM-L6-v2": 0.0,  # runs locally - free
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "voyage-3": 0.06,
}

_UNKNOWN_EMBEDDING_RATE = 0.0  # assume an unlisted embedding model is local / free


def apply_configured_pricing() -> None:
    """Merge overrides from ``settings.pricing_file`` if it exists.

    File shape: ``{"models": {"my-model": {"input_per_million": 1, "output_per_million": 2}},
    "embeddings": {"my-embed": 0.05}}``. Any subset is fine.
    """
    if not settings.pricing_file:
        return
    path = Path(settings.pricing_file)
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for model, rate in data.get("models", {}).items():
        MODEL_PRICING[model] = ModelPrice.model_validate(rate)
    for model, rate in data.get("embeddings", {}).items():
        EMBEDDING_PRICING[model] = float(rate)


def model_price(model: str) -> Optional[ModelPrice]:
    return MODEL_PRICING.get(model)


def embedding_rate_per_million(model: str) -> float:
    return EMBEDDING_PRICING.get(model, _UNKNOWN_EMBEDDING_RATE)


def llm_cost(usage: TokenUsage, model: str) -> float:
    """USD for `usage`'s prompt+completion tokens at `model`'s rate. Unknown model -> 0."""
    price = MODEL_PRICING.get(model)
    if price is None:
        return 0.0
    return (
        usage.prompt_tokens * price.input_per_million
        + usage.completion_tokens * price.output_per_million
    ) / 1_000_000


def embedding_cost(tokens: int, model: str) -> float:
    """USD for `tokens` embedding tokens at `model`'s rate."""
    return tokens * embedding_rate_per_million(model) / 1_000_000
