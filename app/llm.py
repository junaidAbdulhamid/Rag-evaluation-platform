"""A minimal text-in / text-out LLM interface.

`LLMGenerator` (in `app/generation/`) is RAG-specific: it takes a question plus
retrieved chunks and returns a `GeneratedAnswer`. But from Phase 4 on we also need
plain "send this prompt, get text back" calls for evaluation - the LLM-as-judge, and
later the claim extractor in Phase 5. `TextLLM` is that lower-level seam.

Keeping it separate means:
* the judge can run on a different model than the system under test,
* tests inject a fake that returns canned strings - no network, no key,
* every Claude-client wiring detail lives in exactly one place (`AnthropicTextLLM`),
  which `AnthropicGenerator` now builds on too.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from app.models import TokenUsage


class LLMTextResponse(BaseModel):
    text: str
    model: str
    token_usage: Optional[TokenUsage] = None


class TextLLM(ABC):
    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        effort: str = "low",
    ) -> LLMTextResponse:
        """Send one user prompt, return the model's text plus token usage."""


# Substrings of model ids that accept `output_config.effort`. Opus 4.5+, Sonnet 5,
# and the 4.6+ family support it; Haiku 4.5, Sonnet 4.5, and older reject it with a
# 400. Passing effort is an optimisation, not a requirement, so we just drop it for
# models that can't take it.
_EFFORT_MODEL_MARKERS = (
    "opus-5", "opus-4-8", "opus-4-7", "opus-4-6", "opus-4-5",
    "sonnet-5", "sonnet-4-6", "fable-5", "mythos-5",
)


def _supports_effort(model: str) -> bool:
    return any(marker in model for marker in _EFFORT_MODEL_MARKERS)


class AnthropicTextLLM(TextLLM):
    """The single place the Anthropic client is constructed and called."""

    def __init__(self, model: str = "claude-opus-5", api_key: Optional[str] = None) -> None:
        import anthropic

        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._model = model

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        effort: str = "low",
    ) -> LLMTextResponse:
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if _supports_effort(self._model):
            kwargs["output_config"] = {"effort": effort}
        if system is not None:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )
        return LLMTextResponse(text=text, model=self._model, token_usage=usage)
