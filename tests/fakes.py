"""Test doubles.

These live under `tests/` (not `app/`) on purpose - they are not production code.
They let us unit-test orchestration logic (retriever ranking, pipeline wiring)
without loading a real embedding model or calling a paid API.
"""

from __future__ import annotations

import hashlib
import re

from app.generation.generator import LLMGenerator
from app.ingestion.embeddings import EmbeddingProvider
from app.llm import LLMTextResponse, TextLLM
from app.models import GeneratedAnswer, RetrievedChunk, TokenUsage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedding.

    Not semantically smart, but it has the one property the tests need: texts that
    share words get vectors that point in similar directions. No torch, no network.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._dimension
            vector[bucket] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return "fake-hashing-embedding"


class EchoGenerator(LLMGenerator):
    """LLM stand-in: echoes how many chunks it saw. Records the last call for asserts."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GeneratedAnswer:
        self.calls.append((question, retrieved))
        return GeneratedAnswer(
            answer=f"Answered '{question}' using {len(retrieved)} chunk(s).",
            model="echo-generator",
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            prompt="(fake prompt)",
        )


class FakeTextLLM(TextLLM):
    """Returns scripted responses in order; the last one repeats. Records prompts."""

    def __init__(self, responses: str | list[str]) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[str] = []

    def complete(self, prompt, *, system=None, max_tokens=1024, effort="low") -> LLMTextResponse:
        self.calls.append(prompt)
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return LLMTextResponse(
            text=self._responses[index],
            model="fake-text-llm",
            token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
