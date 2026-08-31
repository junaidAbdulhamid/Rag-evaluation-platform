"""Token metering for LLM calls made during evaluation.

`retry_structured_call` (the judge / faithfulness / citation path) throws away the
`LLMTextResponse` and returns only the parsed model, so those token counts would be
invisible. `RecordingTextLLM` is a transparent `TextLLM` decorator that adds every
call's usage into a shared `TokenMeter`.

Generation tokens are read straight off `GeneratedAnswer.token_usage` /
`CitedAnswer.token_usage`, so the generator's LLM is left *un*-metered - that keeps
generation and evaluation token counts separate and avoids double-counting.
"""

from __future__ import annotations

from typing import Optional

from app.llm import LLMTextResponse, TextLLM
from app.models import TokenUsage


def add_usage(a: TokenUsage, b: Optional[TokenUsage]) -> TokenUsage:
    if b is None:
        return a
    return TokenUsage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        total_tokens=a.total_tokens + b.total_tokens,
    )


class TokenMeter:
    def __init__(self) -> None:
        self.calls = 0
        self.usage = TokenUsage()

    def record(self, usage: Optional[TokenUsage]) -> None:
        if usage is not None:
            self.calls += 1
            self.usage = add_usage(self.usage, usage)

    def snapshot(self) -> TokenUsage:
        return self.usage.model_copy()

    def delta_since(self, before: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.usage.prompt_tokens - before.prompt_tokens,
            completion_tokens=self.usage.completion_tokens - before.completion_tokens,
            total_tokens=self.usage.total_tokens - before.total_tokens,
        )


class RecordingTextLLM(TextLLM):
    def __init__(self, inner: TextLLM, meter: TokenMeter) -> None:
        self._inner = inner
        self._meter = meter

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 1024,
        effort: str = "low",
    ) -> LLMTextResponse:
        response = self._inner.complete(
            prompt, system=system, max_tokens=max_tokens, effort=effort
        )
        self._meter.record(response.token_usage)
        return response
