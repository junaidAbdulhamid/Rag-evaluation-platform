"""LLM generator.

`LLMGenerator` is the abstraction; `AnthropicGenerator` is the Phase 1 implementation.
As of Phase 4 it composes `app.llm.AnthropicTextLLM` rather than talking to the
Anthropic SDK directly - the client wiring, effort setting, and token-usage
extraction now live in one place and are shared with the LLM-as-judge. The public
constructor is unchanged (`model`, `api_key`, `max_tokens`); an optional `llm=`
argument lets tests inject a fake `TextLLM`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.generation.prompt import SYSTEM_PROMPT, build_rag_prompt
from app.llm import AnthropicTextLLM, TextLLM
from app.models import GeneratedAnswer, RetrievedChunk


class LLMGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        """Produce a grounded answer to `question` given `retrieved` context."""


class AnthropicGenerator(LLMGenerator):
    def __init__(
        self,
        model: str = "claude-opus-5",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        *,
        llm: Optional[TextLLM] = None,
    ) -> None:
        self._llm = llm or AnthropicTextLLM(model=model, api_key=api_key)
        self._max_tokens = max_tokens

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        prompt = build_rag_prompt(question, retrieved)
        response = self._llm.complete(
            prompt, system=SYSTEM_PROMPT, max_tokens=self._max_tokens
        )
        return GeneratedAnswer(
            answer=response.text.strip(),
            model=response.model,
            token_usage=response.token_usage,
            prompt=prompt,
        )
