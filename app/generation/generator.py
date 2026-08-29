"""LLM generator.

`LLMGenerator` is the abstraction; `AnthropicGenerator` is the Phase 1 implementation
(Claude via the official `anthropic` SDK). Anything downstream - the pipeline, the
experiment runner, the LLM-as-judge evaluators in Phase 4/5 - depends on the base
class, so the judge could run on a different model than the system under test.

Notes on the Anthropic call:
* `temperature` is intentionally not passed. Current Claude models (Opus 5, Sonnet 5,
  the 4.6+ family) reject sampling parameters; determinism is controlled with
  `effort` instead. A future non-Claude generator can reintroduce temperature.
* `effort="low"` keeps latency and token spend down - grounded extraction from a
  short context is not a hard reasoning task.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.generation.prompt import SYSTEM_PROMPT, build_rag_prompt
from app.models import GeneratedAnswer, RetrievedChunk, TokenUsage


class LLMGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        """Produce a grounded answer to `question` given `retrieved` context."""


class AnthropicGenerator(LLMGenerator):
    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        import anthropic

        # Anthropic() with no api_key falls back to the ANTHROPIC_API_KEY env var
        # or an `ant auth login` profile; pass one explicitly only when we have it.
        self._client = (
            anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        )
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        prompt = build_rag_prompt(question, retrieved)

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )

        answer_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return GeneratedAnswer(
            answer=answer_text,
            model=self._model,
            token_usage=usage,
            prompt=prompt,
        )
