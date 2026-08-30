"""LLM-as-a-judge for generation quality.

The judge reads (question, reference answer, generated answer) and returns two
scores in [0, 1] - ``correctness`` and ``relevance`` - each with a one-sentence
rationale. Output is a strict Pydantic model: the score fields are range-validated,
unknown keys are rejected, and blank reasoning is rejected. The parse + retry loop
is shared with the faithfulness evaluator (see ``structured_output.py``); when it
gives up it raises ``JudgeParseError``.

The judge depends on the ``TextLLM`` seam, so a test injects a fake that returns
canned strings and the retry / parse logic is exercised with no network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.evaluation.structured_output import (
    StructuredOutputError,
    parse_model,
    retry_structured_call,
)
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


class JudgeParseError(StructuredOutputError):
    """Raised when the judge never returns parseable structured output."""


def build_judge_prompt(question: str, expected_answer: str, generated_answer: str) -> str:
    return _JUDGE_TEMPLATE.format(
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )


def parse_judgement(text: str) -> GenerationJudgement:
    """str -> GenerationJudgement, raising JSONDecodeError or ValidationError."""
    return parse_model(text, GenerationJudgement)


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
        return retry_structured_call(
            self._llm,
            prompt=prompt,
            response_model=GenerationJudgement,
            system=JUDGE_SYSTEM,
            max_tokens=600,
            max_retries=self._max_retries,
            error_cls=JudgeParseError,
        )
