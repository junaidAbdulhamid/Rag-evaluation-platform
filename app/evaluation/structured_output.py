"""Shared plumbing for "ask an LLM for JSON, validate it, retry if it's malformed".

By Phase 5 this pattern has three call sites - the generation judge, faithfulness
claim extraction, and faithfulness claim verification - so the retry loop and the
fence/prose-tolerant JSON extractor live here instead of being copy-pasted.
"""

from __future__ import annotations

import json
from typing import Optional, TypeVar

from pydantic import BaseModel, ValidationError

from app.llm import TextLLM

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when an LLM never returns parseable structured output."""


_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed ({error}). "
    "Respond with ONLY the JSON object described above - no prose, no code fences."
)


def extract_json_object(text: str) -> str:
    """Pull the outermost {...} out of a response that may have fences or prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise json.JSONDecodeError("no JSON object found in response", text or "", 0)
    return text[start : end + 1]


def parse_model(text: str, response_model: type[T]) -> T:
    """str -> validated model, raising JSONDecodeError or ValidationError on failure."""
    return response_model.model_validate(json.loads(extract_json_object(text)))


def retry_structured_call(
    llm: TextLLM,
    *,
    prompt: str,
    response_model: type[T],
    system: Optional[str] = None,
    max_retries: int = 2,
    max_tokens: int = 800,
    effort: str = "medium",
    error_cls: type[Exception] = StructuredOutputError,
) -> T:
    """Call ``llm``, parse its text into ``response_model``, retry on failure.

    Each retry appends a short "your last output was invalid" note to the prompt.
    After ``max_retries`` further attempts, raises ``error_cls`` rather than
    returning something unvalidated.
    """
    suffix = ""
    last_error: Exception | None = None

    for _attempt in range(max_retries + 1):
        response = llm.complete(
            prompt + suffix, system=system, max_tokens=max_tokens, effort=effort
        )
        try:
            return parse_model(response.text, response_model)
        except (json.JSONDecodeError, ValidationError) as error:
            last_error = error
            suffix = _RETRY_SUFFIX.format(error=error)

    raise error_cls(
        f"invalid structured output after {max_retries + 1} attempts: {last_error}"
    )
