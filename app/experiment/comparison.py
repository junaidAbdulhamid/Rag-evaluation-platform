"""Compare two or more completed experiments.

The output is a ``ComparisonReport``: the same metrics side by side, direction-aware
deltas against a baseline, a config diff, and a tradeoff summary per experiment.

Design choices the spec calls out:
* deltas carry both absolute and percent, and a ``direction`` (improved / regressed /
  neutral) that already accounts for "lower is better" metrics (latency, cost,
  hallucination rate);
* the report **does not pick a winner** - it lists what each experiment gains and
  loses relative to the baseline and leaves the quality/cost/latency call to a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel

from app.experiment.config import ExperimentConfig
from app.experiment.results import ExperimentResult

# metrics whose deltas are performance tradeoffs, not quality changes
_PERF_KEYS = {
    "latency_total_ms",
    "latency_total_p95_ms",
    "latency_generation_ms",
    "latency_evaluation_ms",
    "cost_per_query_usd",
    "cost_total_usd",
}

_NEUTRAL_PCT = 0.5  # |Δ| below this % counts as noise


def _g(agg: Any, attr: str) -> Optional[float]:
    return getattr(agg, attr, None) if agg is not None else None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    higher_is_better: bool
    unit: str  # "" (score) | "ms" | "$"
    getter: Callable[[ExperimentResult], Optional[float]]


METRICS: list[MetricSpec] = [
    MetricSpec("hit_rate", "Hit Rate", True, "", lambda r: _g(r.retrieval, "hit_rate")),
    MetricSpec("precision", "Precision@K", True, "", lambda r: _g(r.retrieval, "precision")),
    MetricSpec("recall", "Recall@K", True, "", lambda r: _g(r.retrieval, "recall")),
    MetricSpec("mrr", "MRR", True, "", lambda r: _g(r.retrieval, "mrr")),
    MetricSpec("ndcg", "NDCG@K", True, "", lambda r: _g(r.retrieval, "ndcg")),
    MetricSpec("correctness", "Correctness", True, "", lambda r: _g(r.generation, "judge_correctness")),
    MetricSpec("relevance", "Relevance", True, "", lambda r: _g(r.generation, "judge_relevance")),
    MetricSpec("exact_match", "Exact Match", True, "", lambda r: _g(r.generation, "exact_match")),
    MetricSpec("token_f1", "Token F1", True, "", lambda r: _g(r.generation, "token_f1")),
    MetricSpec("faithfulness", "Faithfulness", True, "", lambda r: _g(r.faithfulness, "faithfulness")),
    MetricSpec("citation_precision", "Citation Precision", True, "", lambda r: _g(r.citation, "citation_precision")),
    MetricSpec("citation_completeness", "Citation Completeness", True, "", lambda r: _g(r.citation, "citation_completeness")),
    MetricSpec("citation_correctness", "Citation Correctness", True, "", lambda r: _g(r.citation, "citation_correctness")),
    MetricSpec("citation_hallucination_rate", "Citation Hallucination", False, "", lambda r: _g(r.citation, "citation_hallucination_rate")),
    MetricSpec("latency_total_ms", "Latency total (mean)", False, "ms", lambda r: r.latency.total_ms),
    MetricSpec("latency_total_p95_ms", "Latency total (p95)", False, "ms",
               lambda r: r.latency_report.p95("total") if r.latency_report else None),
    MetricSpec("latency_generation_ms", "Latency generation", False, "ms", lambda r: r.latency.generation_ms),
    MetricSpec("latency_evaluation_ms", "Latency evaluation", False, "ms", lambda r: r.latency.evaluation_ms),
    MetricSpec("cost_per_query_usd", "Cost / query", False, "$", lambda r: r.cost.cost_per_query_usd),
    MetricSpec("cost_total_usd", "Cost total", False, "$", lambda r: r.cost.total_usd),
]


class Delta(BaseModel):
    absolute: float
    percent: Optional[float]  # None when the baseline value is 0
    direction: str            # "improved" | "regressed" | "neutral"


class MetricComparison(BaseModel):
    key: str
    label: str
    unit: str
    higher_is_better: bool
    baseline_id: str
    values: dict[str, Optional[float]]      # experiment_id -> value
    deltas: dict[str, Optional[Delta]]      # experiment_id -> delta vs baseline (None for baseline / missing)


class ConfigDiff(BaseModel):
    field: str
    values: dict[str, Any]                  # experiment_id -> config value


class Tradeoff(BaseModel):
    experiment_id: str
    gains: list[str]
    losses: list[str]
    summary: str


class ComparisonReport(BaseModel):
    experiment_ids: list[str]
    baseline_id: str
    config_diff: list[ConfigDiff]
    metrics: list[MetricComparison]
    tradeoffs: list[Tradeoff]


def _delta(baseline: float, other: float, higher_is_better: bool) -> Delta:
    absolute = other - baseline
    percent = (absolute / abs(baseline) * 100.0) if baseline != 0 else None

    if absolute == 0 or (percent is not None and abs(percent) < _NEUTRAL_PCT):
        direction = "neutral"
    else:
        improved = (absolute > 0) if higher_is_better else (absolute < 0)
        direction = "improved" if improved else "regressed"

    return Delta(
        absolute=round(absolute, 8),
        percent=round(percent, 2) if percent is not None else None,
        direction=direction,
    )


def format_delta(delta: Delta, unit: str) -> str:
    sign = "+" if delta.absolute >= 0 else "-"
    mag = abs(delta.absolute)
    if unit == "ms":
        body = f"{sign}{mag:.0f} ms"
    elif unit == "$":
        body = f"{sign}${mag:.6f}"
    else:
        body = f"{sign}{mag:.4f}"
    if delta.percent is not None:
        body += f" ({sign}{abs(delta.percent):.1f}%)"
    tag = {"improved": "better", "regressed": "worse", "neutral": "~"}[delta.direction]
    return f"{body} [{tag}]"


def _hashable(value: Any) -> Any:
    return tuple(value) if isinstance(value, list) else value


def _config_diff(results: list[ExperimentResult]) -> list[ConfigDiff]:
    diffs: list[ConfigDiff] = []
    for field in ExperimentConfig.model_fields:
        if field == "experiment_name":
            continue
        values = {r.experiment_id: getattr(r.config, field) for r in results}
        if len({_hashable(v) for v in values.values()}) > 1:
            diffs.append(ConfigDiff(field=field, values=values))
    return diffs


def _tradeoffs(
    results: list[ExperimentResult], baseline_id: str, metrics: list[MetricComparison]
) -> list[Tradeoff]:
    out: list[Tradeoff] = []
    for result in results:
        if result.experiment_id == baseline_id:
            continue
        gains: list[str] = []
        losses: list[str] = []
        gain_quality = gain_perf = loss_quality = loss_perf = False

        for mc in metrics:
            delta = mc.deltas.get(result.experiment_id)
            if delta is None or delta.direction == "neutral":
                continue
            text = f"{mc.label} {format_delta(delta, mc.unit)}"
            is_perf = mc.key in _PERF_KEYS
            if delta.direction == "improved":
                gains.append(text)
                gain_perf = gain_perf or is_perf
                gain_quality = gain_quality or not is_perf
            else:
                losses.append(text)
                loss_perf = loss_perf or is_perf
                loss_quality = loss_quality or not is_perf

        if not gains and not losses:
            summary = "no material differences from the baseline"
        elif not losses:
            summary = "strictly better than the baseline on measured metrics"
        elif not gains:
            summary = "strictly worse than the baseline on measured metrics"
        elif gain_quality and loss_perf and not loss_quality:
            summary = "higher quality, but slower and/or more expensive"
        elif gain_perf and loss_quality and not gain_quality:
            summary = "faster and/or cheaper, but lower quality"
        else:
            summary = "mixed changes across quality and performance"

        out.append(
            Tradeoff(experiment_id=result.experiment_id, gains=gains, losses=losses, summary=summary)
        )
    return out


def compare_experiments(
    results: list[ExperimentResult], *, baseline_index: int = 0
) -> ComparisonReport:
    if len(results) < 2:
        raise ValueError("need at least two experiments to compare")
    ids = [r.experiment_id for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate experiment ids in comparison")

    baseline_id = ids[baseline_index]

    metrics: list[MetricComparison] = []
    for spec in METRICS:
        values = {r.experiment_id: spec.getter(r) for r in results}
        base_val = values[baseline_id]
        deltas: dict[str, Optional[Delta]] = {}
        for r in results:
            if r.experiment_id == baseline_id:
                deltas[r.experiment_id] = None
            elif base_val is None or values[r.experiment_id] is None:
                deltas[r.experiment_id] = None
            else:
                deltas[r.experiment_id] = _delta(
                    base_val, values[r.experiment_id], spec.higher_is_better
                )
        metrics.append(
            MetricComparison(
                key=spec.key,
                label=spec.label,
                unit=spec.unit,
                higher_is_better=spec.higher_is_better,
                baseline_id=baseline_id,
                values=values,
                deltas=deltas,
            )
        )

    return ComparisonReport(
        experiment_ids=ids,
        baseline_id=baseline_id,
        config_diff=_config_diff(results),
        metrics=metrics,
        tradeoffs=_tradeoffs(results, baseline_id, metrics),
    )
