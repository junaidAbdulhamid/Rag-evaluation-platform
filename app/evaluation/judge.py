"""LLM-as-a-judge for generation quality.

The judge reads (question, reference answer, generated answer) and returns two
scores in [0, 1] - ``correctness`` and ``relevance`` - each with a one-sentence
rationale. Output is a strict Pydantic model: the score fields are range-validated,
unknown keys are rejected, and blank reasoning is rejected. Anything that fails to
parse triggers a retry with a "your last output was invalid" nudge; after
``max_retries`` the call raises ``JudgeParseError`` rather than returning garbage.

The judge depends on the ``TextLLM`` seam, so a test injects a fake that returns
canned strings and the retry / parse logic is exercised with no network.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.llm import TextLLM

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a question-answering system. You compare a "
    "generated answer against a reference answer and the user's question and output "
    "numeric scores. Respond with a single JSON object and nothing else."
)

_JUDGE_TEMPLATE = """Evaluate the GENERATED ANSWER on two axes.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{expected_answer}

GENERATED ANSWER (to be judged):
{generated_answer}

Scoring guidance:
- correctness (0.0-1.0): does the generated answer convey the same factual content
  as the reference? Judge meaning, not wording. 1.0 = fully correct and complete,
  0.5 = partially correct or missing a key detail, 0.0 = wrong or contradictory.
  If the reference indicates the question cannot be answered from the available
  context, then a generated answer that also declines is correct (1.0) and one that
  invents an answer is incorrect (0.0).
- relevance (0.0-1.0): does the generated answer directly address the question that
  was asked, regardless of correctness? 1.0 = on topic, 0.0 = off topic or evasive.

Respond with ONLY this JSON object, no code fences, no extra text:
{{"correctness": <float 0-1>, "relevance": <float 0-1>, "correctness_reasoning": "<one sentence>", "relevance_reasoning": "<one sentence>"}}"""

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed ({error}). "
    "Respond with ONLY the JSON object described above - no prose, no code fences."
)


class GenerationJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correctness: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    correctness_reasoning: str
    relevance_reasoning: str

    @field_validator("correctness_reasoning", "relevance_reasoning")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("reasoning must not be blank")
        return value.strip()


class JudgeParseError(RuntimeError):
    """Raised when the judge never returns parseable structured output."""


def build_judge_prompt(question: str, expected_answer: str, generated_answer: str) -> str:
    return _JUDGE_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )


def _extract_json_object(text: str) -> str:
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


def parse_judgement(text: str) -> GenerationJudgement:
    """str -> GenerationJudgement, raising JSONDecodeError or ValidationError."""
    return GenerationJudgement.model_validate(json.loads(_extract_json_object(text)))


class GenerationJudge(ABC):
    @abstractmethod
    def judge(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> GenerationJudgement:
        """Score one generated answer."""


class LLMGenerationJudge(GenerationJudge):
    def __init__(self, llm: TextLLM, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def judge(
        self, *, question: str, expected_answer: str, generated_answer: str
    ) -> GenerationJudgement:
        prompt = build_judge_prompt(question, expected_answer, generated_answer)
        suffix = ""
        last_error: Exception | None = None

        for _attempt in range(self._max_retries + 1):
            response = self._llm.complete(
                prompt + suffix, system=JUDGE_SYSTEM, max_tokens=600, effort="medium"
            )
            try:
                return parse_judgement(response.text)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = error
                suffix = _RETRY_SUFFIX.format(error=error)

        raise JudgeParseError(
            f"judge produced invalid output after {self._max_retries + 1} attempts: {last_error}"
        )
