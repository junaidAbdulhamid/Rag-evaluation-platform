"""Citation-grounded generation.

A normal `AnthropicGenerator` produces prose. `AnthropicCitedGenerator` produces
prose with inline `[n]` markers, where `n` is a context block number - the same
1-based numbering `format_context` already puts in the prompt, which equals the
retrieved chunk's `rank`. That is the whole "maintain the connection between citation
numbers and actual retrieved chunks" requirement: marker `n` -> `retrieved[n - 1]`.

Marker resolution is a deterministic post-processing step (`parse_citations`), not
something the model returns as JSON - the model just writes `[1]`, `[2]`, and we
regex them out and look them up.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from app.generation.prompt import INSUFFICIENT_CONTEXT_REPLY, format_context
from app.llm import AnthropicTextLLM, TextLLM
from app.models import Citation, CitedAnswer, RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")

CITED_SYSTEM_PROMPT = f"""You are a precise question-answering assistant that cites its sources.

Answer using ONLY the information in the provided numbered context blocks.

Rules:
- After each sentence that states a fact from the context, append the citation
  marker(s) for the block(s) that support it, e.g. "Refunds take 30 days. [1]".
- A sentence may carry more than one marker: "... [1][3]".
- Use ONLY the block numbers shown in the context. Never invent a number.
- Do not use outside knowledge.
- If the context does not contain the answer, reply with exactly this sentence and
  nothing else: "{INSUFFICIENT_CONTEXT_REPLY}"
- After the answer, add a "Sources:" line listing each marker you used and its source.
"""


def build_cited_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    return (
        f"Context:\n{format_context(retrieved)}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with [n] citation markers):"
    )


def parse_citations(text: str, retrieved: Sequence[RetrievedChunk]) -> list[Citation]:
    """Extract distinct `[n]` markers from `text` and resolve each against `retrieved`."""
    markers = sorted({int(m) for m in _MARKER_RE.findall(text)})
    citations: list[Citation] = []
    for marker in markers:
        exists = 1 <= marker <= len(retrieved)
        chunk = retrieved[marker - 1].chunk if exists else None
        citations.append(
            Citation(
                marker=marker,
                exists=exists,
                chunk_id=chunk.chunk_id if chunk else None,
                document_id=chunk.document_id if chunk else None,
            )
        )
    return citations


def build_cited_answer(
    text: str,
    model: str,
    token_usage,
    prompt: str,
    retrieved: Sequence[RetrievedChunk],
) -> CitedAnswer:
    return CitedAnswer(
        answer=text.strip(),
        citations=parse_citations(text, retrieved),
        model=model,
        token_usage=token_usage,
        prompt=prompt,
    )


class CitedGenerator(ABC):
    @abstractmethod
    def generate(
        self, question: str, retrieved: list[RetrievedChunk]
    ) -> CitedAnswer:
        """Produce a grounded answer with inline citation markers."""


class AnthropicCitedGenerator(CitedGenerator):
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

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> CitedAnswer:
        prompt = build_cited_prompt(question, retrieved)
        response = self._llm.complete(
            prompt, system=CITED_SYSTEM_PROMPT, max_tokens=self._max_tokens
        )
        return build_cited_answer(
            response.text, response.model, response.token_usage, prompt, retrieved
        )
