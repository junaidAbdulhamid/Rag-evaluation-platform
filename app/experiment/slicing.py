"""Per-slice metrics.

Every ``EvalExample`` carries slice labels (``numerical``, ``multi_document``,
``unanswerable``, ...). Phase 14 groups the per-question results by label and runs
the same Phase 3-6 ``aggregate_*`` functions on each group, so you can see that a
system is 0.95 overall but 0.71 on ``multi_document``.

Read-only: a pure function of a stored ``ExperimentResult``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic import BaseModel

from app.evaluation.citation_eval import AggregateCitationMetrics, aggregate_citation_metrics
from app.evaluation.faithfulness_eval import (
    AggregateFaithfulnessMetrics,
    aggregate_faithfulness_metrics,
)
from app.evaluation.generation import AggregateGenerationMetrics, aggregate_generation_metrics
from app.evaluation.retrieval import AggregateRetrievalMetrics, aggregate_retrieval_metrics
from app.experiment.results import ExperimentResult, QuestionExperimentResult

OVERALL = "overall"

# metric name -> (section, attribute); all higher-is-better
_QUALITY_METRICS = {
    "hit_rate": ("retrieval", "hit_rate"),
    "precision": ("retrieval", "precision"),
    "recall": ("retrieval", "recall"),
    "mrr": ("retrieval", "mrr"),
    "ndcg": ("retrieval", "ndcg"),
    "correctness": ("generation", "judge_correctness"),
    "relevance": ("generation", "judge_relevance"),
    "exact_match": ("generation", "exact_match"),
    "token_f1": ("generation", "token_f1"),
    "faithfulness": ("faithfulness", "faithfulness"),
    "citation_precision": ("citation", "citation_precision"),
    "citation_completeness": ("citation", "citation_completeness"),
}


class SliceMetrics(BaseModel):
    label: str
    num_questions: int          # questions carrying this label (errors included)
    num_scored: int             # non-errored questions
    retrieval: Optional[AggregateRetrievalMetrics] = None
    generation: Optional[AggregateGenerationMetrics] = None
    faithfulness: Optional[AggregateFaithfulnessMetrics] = None
    citation: Optional[AggregateCitationMetrics] = None
    latency_total_ms: float = 0.0   # mean
    cost_per_query_usd: float = 0.0  # mean

    def metric(self, name: str) -> Optional[float]:
        if name not in _QUALITY_METRICS:
            return None
        section, attr = _QUALITY_METRICS[name]
        agg = getattr(self, section, None)
        return getattr(agg, attr, None) if agg is not None else None


class SliceUnderperformance(BaseModel):
    label: str
    metric: str
    slice_value: float
    overall_value: float
    gap: float  # slice_value - overall_value (negative == worse)


class SliceReport(BaseModel):
    experiment_id: str
    overall: SliceMetrics
    slices: list[SliceMetrics]


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 4) if values else 0.0


def _slice_metrics(label: str, questions: list[QuestionExperimentResult]) -> SliceMetrics:
    ok = [q for q in questions if q.ok]
    retr = [q.retrieval for q in ok if q.retrieval is not None]
    gen = [q.generation for q in ok if q.generation is not None]
    faith = [q.faithfulness for q in ok if q.faithfulness is not None]
    cite = [q.citation for q in ok if q.citation is not None]

    return SliceMetrics(
        label=label,
        num_questions=len(questions),
        num_scored=len(ok),
        # retrieval metrics are undefined for a slice with no answerable questions
        retrieval=(
            aggregate_retrieval_metrics(retr)
            if any(not r.is_unanswerable for r in retr)
            else None
        ),
        generation=aggregate_generation_metrics(gen) if gen else None,
        faithfulness=aggregate_faithfulness_metrics(faith) if faith else None,
        citation=aggregate_citation_metrics(cite) if cite else None,
        latency_total_ms=_mean(q.latency_ms.get("total", 0.0) for q in ok),
        cost_per_query_usd=_mean(q.cost.total_usd for q in ok),
    )


def slice_report(result: ExperimentResult) -> SliceReport:
    per_q = result.per_question
    labels = sorted({label for q in per_q for label in q.slices})
    return SliceReport(
        experiment_id=result.experiment_id,
        overall=_slice_metrics(OVERALL, per_q),
        slices=[
            _slice_metrics(label, [q for q in per_q if label in q.slices]) for label in labels
        ],
    )


def underperforming_slices(
    report: SliceReport, *, metric: str = "recall", min_gap: float = 0.05
) -> list[SliceUnderperformance]:
    """Slices whose `metric` is at least `min_gap` below the overall value."""
    overall_value = report.overall.metric(metric)
    if overall_value is None:
        return []
    out: list[SliceUnderperformance] = []
    for sm in report.slices:
        value = sm.metric(metric)
        if value is not None and value < overall_value - min_gap:
            out.append(
                SliceUnderperformance(
                    label=sm.label,
                    metric=metric,
                    slice_value=round(value, 4),
                    overall_value=round(overall_value, 4),
                    gap=round(value - overall_value, 4),
                )
            )
    return sorted(out, key=lambda u: u.gap)
