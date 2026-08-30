"""Faithfulness evaluation.

Faithfulness is **reference-free**: it does not look at the golden answer at all. It
asks only "is every factual claim in the generated answer supported by the retrieved
context?" You can be faithful but wrong (grounded in a context chunk that is itself
irrelevant) or correct but unfaithful (right answer, but the model used outside
knowledge the context never provided). Phase 4 measures the second thing;
faithfulness measures grounding.

Two LLM steps (the spec's pipeline):

    answer -> [extract] -> atomic claims -> [verify vs numbered context] -> per-claim verdict
           -> faithfulness score = supported claims / total claims

``supporting_chunk_ids`` links each claim back to the actual retrieved chunks: the
verifier works with 1-based block numbers ([1], [2], ... - the same numbering
``prompt.py`` already puts in the RAG prompt), and this module maps those back to
``RetrievedChunk.chunk.chunk_id``. Phase 6 (citations) reuses that idea.

**No factual claims** (a pure abstention, "I don't know", a question): extraction
returns an empty list, ``score`` is ``None``, and the aggregate skips it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from app.evaluation.structured_output import retry_structured_call
from app.generation.prompt import format_context
from app.llm import TextLLM
from app.models import RetrievedChunk

# ---------------------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------------------


class Claim(BaseModel):
    text: str
    supported: bool
    supporting_chunk_ids: list[str] = []
    reason: str


class FaithfulnessResult(BaseModel):
    claims: list[Claim]
    num_claims: int
    num_supported: int
    score: Optional[float]  # supported / total; None when there are no factual claims

    @classmethod
    def from_claims(cls, claims: list[Claim]) -> "FaithfulnessResult":
        num_claims = len(claims)
        num_supported = sum(1 for claim in claims if claim.supported)
        return cls(
            claims=claims,
            num_claims=num_claims,
            num_supported=num_supported,
            score=(num_supported / num_claims) if num_claims else None,
        )

    @classmethod
    def no_claims(cls) -> "FaithfulnessResult":
        return cls(claims=[], num_claims=0, num_supported=0, score=None)


# ---------------------------------------------------------------------------------------
# LLM I/O models (what we ask the model to return at each step)
# ---------------------------------------------------------------------------------------


class _ClaimList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[str]

    @field_validator("claims")
    @classmethod
    def _clean(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item and item.strip()]


class _Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_index: int
    supported: bool
    supporting_blocks: list[int] = []
    reason: str


class _VerdictList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdicts: list[_Verdict]


# ---------------------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------------------

_EXTRACT_SYSTEM = (
    "You extract atomic factual claims from text. Respond with a single JSON object "
    "and nothing else."
)

_EXTRACT_TEMPLATE = """Break the ANSWER below into a list of atomic, standalone factual claims.

Rules:
- Each claim states exactly one fact and is understandable on its own (resolve pronouns).
- Include only verifiable factual statements. Exclude questions, opinions, hedges, and
  statements that merely say the context is insufficient or that no answer is available.
- If the answer contains no factual claims, return an empty list.

ANSWER:
{answer}

Respond with ONLY this JSON object:
{{"claims": ["<claim 1>", "<claim 2>", ...]}}"""

_VERIFY_SYSTEM = (
    "You check whether each claim is supported by the given context. Respond with a "
    "single JSON object and nothing else."
)

_VERIFY_TEMPLATE = """For each CLAIM, decide whether it is directly supported by the CONTEXT blocks.
"Supported" means the context explicitly states or clearly entails the claim. If the
context is silent, contradicts it, or only partially supports it, mark it unsupported.

CONTEXT:
{context}

CLAIMS:
{claims}

For every claim index, output one verdict object with:
- "claim_index": the integer index shown above
- "supported": true or false
- "supporting_blocks": list of CONTEXT block numbers (integers) that support the claim
  (empty when unsupported)
- "reason": one sentence

Respond with ONLY this JSON object, one verdict per claim, in index order:
{{"verdicts": [{{"claim_index": 0, "supported": true, "supporting_blocks": [1], "reason": "..."}}]}}"""


def build_extract_prompt(answer: str) -> str:
    return _EXTRACT_TEMPLATE.format(answer=answer)


def build_verify_prompt(claims: Sequence[str], retrieved: Sequence[RetrievedChunk]) -> str:
    numbered_claims = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims))
    return _VERIFY_TEMPLATE.format(context=format_context(retrieved), claims=numbered_claims)


def blocks_to_chunk_ids(
    blocks: Sequence[int], retrieved: Sequence[RetrievedChunk]
) -> list[str]:
    """Map 1-based context block numbers to chunk ids, dropping out-of-range numbers."""
    chunk_ids = []
    for block in blocks:
        if 1 <= block <= len(retrieved):
            chunk_ids.append(retrieved[block - 1].chunk.chunk_id)
    return chunk_ids


# ---------------------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------------------


class FaithfulnessEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, *, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> FaithfulnessResult:
        """Score how well ``answer`` is grounded in ``retrieved``."""


class LLMFaithfulnessEvaluator(FaithfulnessEvaluator):
    def __init__(self, llm: TextLLM, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def evaluate(
        self, *, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> FaithfulnessResult:
        claim_texts = self._extract_claims(answer)
        if not claim_texts:
            return FaithfulnessResult.no_claims()

        verdicts_by_index = self._verify_claims(claim_texts, retrieved)

        claims: list[Claim] = []
        for index, text in enumerate(claim_texts):
            verdict = verdicts_by_index.get(index)
            if verdict is None:
                # verifier dropped this claim - treat as unsupported rather than retry
                claims.append(
                    Claim(text=text, supported=False, reason="no verdict returned by the verifier")
                )
                continue
            claims.append(
                Claim(
                    text=text,
                    supported=verdict.supported,
                    supporting_chunk_ids=blocks_to_chunk_ids(
                        verdict.supporting_blocks if verdict.supported else [], retrieved
                    ),
                    reason=verdict.reason,
                )
            )
        return FaithfulnessResult.from_claims(claims)

    def _extract_claims(self, answer: str) -> list[str]:
        parsed = retry_structured_call(
            self._llm,
            prompt=build_extract_prompt(answer),
            response_model=_ClaimList,
            system=_EXTRACT_SYSTEM,
            max_tokens=600,
            max_retries=self._max_retries,
        )
        return parsed.claims

    def _verify_claims(
        self, claim_texts: Sequence[str], retrieved: Sequence[RetrievedChunk]
    ) -> dict[int, _Verdict]:
        parsed = retry_structured_call(
            self._llm,
            prompt=build_verify_prompt(claim_texts, retrieved),
            response_model=_VerdictList,
            system=_VERIFY_SYSTEM,
            max_tokens=1200,
            max_retries=self._max_retries,
        )
        # index by claim_index; last write wins if the model repeats one
        return {v.claim_index: v for v in parsed.verdicts}
