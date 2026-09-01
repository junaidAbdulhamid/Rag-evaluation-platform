"""Failure analysis - why did questions fail, and which were worst?

Runs entirely off a stored ``ExperimentResult`` (no API calls). For each question it
picks the *primary* failure category, and it builds "worst N" leaderboards.

Category is decided in priority order (a question can trip several checks; the first
that matches wins, because an earlier-stage failure explains the later ones):

    ERROR                a runtime exception was recorded
    INSUFFICIENT_CONTEXT unanswerable question, system correctly declined  (NOT a failure)
    HALLUCINATION        answered an unanswerable question, OR answer has unsupported claims
    RETRIEVAL_FAILURE    the relevant document was never retrieved
    CITATION_FAILURE     answer is plausibly correct but its citations don't hold up
    GENERATION_FAILURE   context was retrieved but the answer is still wrong
    OK                   nothing tripped
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.experiment.results import ExperimentResult, QuestionExperimentResult


class FailureCategory(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
    GENERATION_FAILURE = "GENERATION_FAILURE"
    HALLUCINATION = "HALLUCINATION"
    CITATION_FAILURE = "CITATION_FAILURE"


_NON_FAILURE = {FailureCategory.OK, FailureCategory.INSUFFICIENT_CONTEXT}


class FailureThresholds(BaseModel):
    correctness_min: float = 0.5        # judge correctness below this -> generation failed
    faithfulness_min: float = 0.7       # faithfulness below this -> hallucination
    citation_precision_min: float = 0.999
    citation_ok_correctness: float = 0.5  # "answer may be correct" bar for CITATION_FAILURE


class QuestionDiagnosis(BaseModel):
    question_id: str
    question: str
    category: FailureCategory
    is_failure: bool
    reason: str
    signals: dict = Field(default_factory=dict)


class RankedQuestion(BaseModel):
    question_id: str
    question: str
    value: float
    detail: str = ""


class FailureAnalysis(BaseModel):
    experiment_id: str
    num_questions: int
    thresholds: FailureThresholds
    diagnoses: list[QuestionDiagnosis]
    category_counts: dict[str, int]

    lowest_recall: list[RankedQuestion]
    lowest_faithfulness: list[RankedQuestion]
    lowest_correctness: list[RankedQuestion]
    highest_latency: list[RankedQuestion]
    highest_cost: list[RankedQuestion]

    retrieval_failures: list[str]
    generation_failures: list[str]      # retrieved OK but wrong
    hallucinations: list[str]
    citation_failures: list[str]

    @property
    def failures(self) -> list[QuestionDiagnosis]:
        return [d for d in self.diagnoses if d.is_failure]


# --- per-question diagnosis ----------------------------------------------------------


def _signals(q: QuestionExperimentResult) -> dict:
    sig: dict = {}
    if q.retrieval is not None:
        m = q.retrieval.metrics
        sig.update(hit_rate=m.hit_rate, recall=m.recall, mrr=m.reciprocal_rank)
    if q.generation is not None:
        if q.generation.judgement is not None:
            sig["correctness"] = q.generation.judgement.correctness
            sig["relevance"] = q.generation.judgement.relevance
        d = q.generation.deterministic
        sig.update(exact_match=d.exact_match, token_f1=d.token_f1, abstained=d.abstained)
    if q.faithfulness is not None:
        sig["faithfulness"] = q.faithfulness.result.score
    if q.citation is not None:
        cr = q.citation.result
        sig.update(
            citation_precision=cr.citation_precision,
            citation_completeness=cr.citation_completeness,
            hallucinated_markers=cr.num_hallucinated_links,
        )
    sig["latency_total_ms"] = q.latency_ms.get("total")
    sig["cost_usd"] = q.cost.total_usd
    return sig


def diagnose_question(
    q: QuestionExperimentResult, thresholds: FailureThresholds
) -> QuestionDiagnosis:
    sig = _signals(q)

    def d(cat: FailureCategory, reason: str) -> QuestionDiagnosis:
        return QuestionDiagnosis(
            question_id=q.question_id,
            question=q.question,
            category=cat,
            is_failure=cat not in _NON_FAILURE,
            reason=reason,
            signals=sig,
        )

    if q.error:
        return d(FailureCategory.ERROR, f"runtime error: {q.error}")

    retr = q.retrieval
    gen = q.generation

    unanswerable = (
        retr.is_unanswerable
        if retr is not None
        else (gen.deterministic.abstention_expected if gen is not None else False)
    )
    abstained = gen.deterministic.abstained if gen is not None else False

    if unanswerable:
        if abstained:
            return d(FailureCategory.INSUFFICIENT_CONTEXT, "correctly declined an unanswerable question")
        return d(FailureCategory.HALLUCINATION, "invented an answer to an unanswerable question")

    # --- answerable from here ---
    if retr is not None and retr.metrics.hit_rate == 0.0:
        return d(
            FailureCategory.RETRIEVAL_FAILURE,
            f"relevant docs {retr.relevant_doc_ids} not retrieved (got {retr.retrieved_doc_ids})",
        )

    faith_score = (
        q.faithfulness.result.score
        if (q.faithfulness is not None and q.faithfulness.result.score is not None)
        else None
    )
    if faith_score is not None and faith_score < thresholds.faithfulness_min:
        r = q.faithfulness.result
        return d(
            FailureCategory.HALLUCINATION,
            f"faithfulness={faith_score:.2f} ({r.num_claims - r.num_supported}/{r.num_claims} claims unsupported)",
        )

    correctness = gen.judgement.correctness if (gen is not None and gen.judgement is not None) else None

    if q.citation is not None:
        cr = q.citation.result
        citation_bad = cr.num_hallucinated_links > 0 or (
            cr.citation_precision is not None
            and cr.citation_precision < thresholds.citation_precision_min
        )
        answer_plausible = correctness is None or correctness >= thresholds.citation_ok_correctness
        if citation_bad and answer_plausible:
            return d(
                FailureCategory.CITATION_FAILURE,
                f"citation precision={cr.citation_precision}, "
                f"{cr.num_hallucinated_links} hallucinated marker(s)",
            )

    if correctness is not None and correctness < thresholds.correctness_min:
        recall = retr.metrics.recall if retr is not None else None
        return d(
            FailureCategory.GENERATION_FAILURE,
            f"context retrieved (recall={recall}) but correctness={correctness:.2f}",
        )

    return d(FailureCategory.OK, "no failure detected")


# --- leaderboards -------------------------------------------------------------------


def _rank(
    questions: list[QuestionExperimentResult],
    value_of: Callable[[QuestionExperimentResult], Optional[float]],
    *,
    top_n: int,
    largest: bool,
    detail_of: Optional[Callable[[QuestionExperimentResult], str]] = None,
) -> list[RankedQuestion]:
    rows = [(q, v) for q in questions if (v := value_of(q)) is not None]
    rows.sort(key=lambda t: t[1], reverse=largest)
    return [
        RankedQuestion(
            question_id=q.question_id,
            question=q.question,
            value=round(float(v), 4),
            detail=detail_of(q) if detail_of else "",
        )
        for q, v in rows[:top_n]
    ]


def analyze_failures(
    result: ExperimentResult,
    *,
    thresholds: Optional[FailureThresholds] = None,
    top_n: int = 5,
) -> FailureAnalysis:
    thresholds = thresholds or FailureThresholds()
    per_q = result.per_question
    diagnoses = [diagnose_question(q, thresholds) for q in per_q]

    counts: dict[str, int] = {}
    for d in diagnoses:
        counts[d.category.value] = counts.get(d.category.value, 0) + 1

    by_cat: dict[FailureCategory, list[str]] = {}
    for d in diagnoses:
        by_cat.setdefault(d.category, []).append(d.question_id)

    def recall_of(q):
        return (
            q.retrieval.metrics.recall
            if (q.retrieval is not None and not q.retrieval.is_unanswerable)
            else None
        )

    return FailureAnalysis(
        experiment_id=result.experiment_id,
        num_questions=len(per_q),
        thresholds=thresholds,
        diagnoses=diagnoses,
        category_counts=counts,
        lowest_recall=_rank(
            per_q, recall_of, top_n=top_n, largest=False,
            detail_of=lambda q: f"retrieved={q.retrieved_doc_ids}",
        ),
        lowest_faithfulness=_rank(
            per_q,
            lambda q: q.faithfulness.result.score
            if (q.faithfulness is not None and q.faithfulness.result.score is not None)
            else None,
            top_n=top_n, largest=False,
        ),
        lowest_correctness=_rank(
            per_q,
            lambda q: q.generation.judgement.correctness
            if (q.generation is not None and q.generation.judgement is not None)
            else None,
            top_n=top_n, largest=False,
        ),
        highest_latency=_rank(
            per_q, lambda q: q.latency_ms.get("total"), top_n=top_n, largest=True
        ),
        highest_cost=_rank(
            per_q, lambda q: q.cost.total_usd, top_n=top_n, largest=True
        ),
        retrieval_failures=by_cat.get(FailureCategory.RETRIEVAL_FAILURE, []),
        generation_failures=by_cat.get(FailureCategory.GENERATION_FAILURE, []),
        hallucinations=by_cat.get(FailureCategory.HALLUCINATION, []),
        citation_failures=by_cat.get(FailureCategory.CITATION_FAILURE, []),
    )
