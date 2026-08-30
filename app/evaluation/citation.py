"""Citation evaluation.

Given a cited answer (prose with inline `[n]` markers) and the retrieved chunks,
score four things the Phase 6 spec asks for:

* **completeness** - of the factual claims in the answer, how many carry a citation?
  (In a citation-grounded system every factual claim should.)
* **precision** - of the citation links included, how many actually support their
  claim? Hallucinated markers count against this.
* **correctness** - per cited claim, is it backed by at least one citation whose
  source really supports it?
* **hallucination rate** - fraction of citation links whose marker points outside
  the retrieved set.

Vocabulary: a **link** is one (claim, marker) pair. A claim that cites `[1][3]`
produces two links.

One LLM call does the work: it segments the answer into atomic claims, reports the
markers attached to each, and - given the numbered context - reports which of those
markers' blocks actually support the claim. Marker *existence* (is `n` in 1..k?) and
all the arithmetic are then deterministic here.

No factual claims (an abstention): every metric is ``None`` and the aggregate skips
the question.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from app.evaluation.structured_output import retry_structured_call
from app.generation.prompt import format_context
from app.llm import TextLLM
from app.models import RetrievedChunk

_MARKER_RE = re.compile(r"\[(\d+)\]")

# ---------------------------------------------------------------------------------------
# Result models  ("store detailed citation evaluation results")
# ---------------------------------------------------------------------------------------


class CitedClaim(BaseModel):
    text: str
    markers: list[int]          # markers attached to this claim, reconciled with the text
    has_citation: bool


class CitationLink(BaseModel):
    claim_index: int
    claim_text: str
    marker: int
    exists: bool                       # marker resolves to a real retrieved chunk
    resolved_chunk_id: Optional[str]
    supports_claim: Optional[bool]     # None when the marker does not exist


class CitationEvaluationResult(BaseModel):
    claims: list[CitedClaim]
    links: list[CitationLink]

    num_claims: int
    num_claims_with_citation: int
    num_citation_links: int
    num_existing_links: int
    num_hallucinated_links: int
    num_supported_links: int

    citation_completeness: Optional[float]      # claims_with_citation / claims
    citation_precision: Optional[float]         # supported_links / all_links
    citation_correctness: Optional[float]       # correctly-cited claims / cited claims
    citation_hallucination_rate: Optional[float]  # hallucinated_links / all_links

    @classmethod
    def compute(
        cls, claims: list[CitedClaim], links: list[CitationLink]
    ) -> "CitationEvaluationResult":
        n_claims = len(claims)
        cited_claims = [c for c in claims if c.has_citation]
        n_links = len(links)
        existing = [link for link in links if link.exists]
        hallucinated = [link for link in links if not link.exists]
        supported = [link for link in existing if link.supports_claim]

        correctly_cited = {link.claim_index for link in existing if link.supports_claim}
        n_correct_claims = sum(
            1 for idx, c in enumerate(claims) if c.has_citation and idx in correctly_cited
        )

        return cls(
            claims=claims,
            links=links,
            num_claims=n_claims,
            num_claims_with_citation=len(cited_claims),
            num_citation_links=n_links,
            num_existing_links=len(existing),
            num_hallucinated_links=len(hallucinated),
            num_supported_links=len(supported),
            citation_completeness=(len(cited_claims) / n_claims) if n_claims else None,
            citation_precision=(len(supported) / n_links) if n_links else None,
            citation_correctness=(
                n_correct_claims / len(cited_claims) if cited_claims else None
            ),
            citation_hallucination_rate=(
                len(hallucinated) / n_links if n_links else None
            ),
        )


# ---------------------------------------------------------------------------------------
# LLM I/O model
# ---------------------------------------------------------------------------------------


class _ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    markers: list[int] = []
    supported_markers: list[int] = []

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _supported_is_subset(self) -> "_ClaimAssessment":
        self.supported_markers = [m for m in self.supported_markers if m in self.markers]
        return self


class _ClaimAssessmentList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[_ClaimAssessment]


# ---------------------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------------------

_SYSTEM = (
    "You analyse citations in an answer. Respond with a single JSON object and "
    "nothing else."
)

_TEMPLATE = """You are given a numbered CONTEXT and an ANSWER that cites it with [n] markers.

CONTEXT:
{context}

ANSWER:
{answer}

Break the ANSWER into atomic, standalone factual claims. For each claim report:
- "text": the claim, understandable on its own
- "markers": the [n] numbers attached to that claim in the ANSWER (empty list if none)
- "supported_markers": the subset of "markers" whose CONTEXT block *actually* states
  or clearly entails the claim (empty list if none of them do, or if there are none)

Exclude questions, hedges, and statements that merely decline to answer. If the
ANSWER contains no factual claims, return an empty list.

Respond with ONLY this JSON object:
{{"claims": [{{"text": "<claim>", "markers": [<int>], "supported_markers": [<int>]}}]}}"""


def build_citation_prompt(answer: str, retrieved: Sequence[RetrievedChunk]) -> str:
    return _TEMPLATE.format(context=format_context(retrieved), answer=answer)


# ---------------------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------------------


class CitationEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, *, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> CitationEvaluationResult:
        """Score the citations in `answer` against `retrieved`."""


class LLMCitationEvaluator(CitationEvaluator):
    def __init__(self, llm: TextLLM, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    def evaluate(
        self, *, answer: str, retrieved: Sequence[RetrievedChunk]
    ) -> CitationEvaluationResult:
        text_markers = {int(m) for m in _MARKER_RE.findall(answer)}
        n_ctx = len(retrieved)

        parsed = retry_structured_call(
            self._llm,
            prompt=build_citation_prompt(answer, retrieved),
            response_model=_ClaimAssessmentList,
            system=_SYSTEM,
            max_tokens=1500,
            max_retries=self._max_retries,
        )

        claims: list[CitedClaim] = []
        links: list[CitationLink] = []
        for claim_index, assessment in enumerate(parsed.claims):
            # trust the text: keep only markers that really appear in the answer
            markers = sorted({m for m in assessment.markers if m in text_markers})
            supported = {m for m in assessment.supported_markers if m in markers}
            claims.append(
                CitedClaim(text=assessment.text, markers=markers, has_citation=bool(markers))
            )
            for marker in markers:
                exists = 1 <= marker <= n_ctx
                links.append(
                    CitationLink(
                        claim_index=claim_index,
                        claim_text=assessment.text,
                        marker=marker,
                        exists=exists,
                        resolved_chunk_id=(
                            retrieved[marker - 1].chunk.chunk_id if exists else None
                        ),
                        supports_claim=(marker in supported) if exists else None,
                    )
                )

        return CitationEvaluationResult.compute(claims, links)
