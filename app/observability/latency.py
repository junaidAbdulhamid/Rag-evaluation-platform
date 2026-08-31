"""Fine-grained latency: per-stage timing + distribution stats.

Phase 7/9 timed the coarse stages (retrieval, generation, evaluation, total) into a
plain dict. Phase 10 wants ``embedding`` and ``retrieval`` split, a ``reranking``
slot, and mean / median / p95 across a run.

Splitting embedding out means timing *inside* ``DenseRetriever.retrieve()`` without
adding a timing argument to the ``BaseRetriever`` interface. The tool for that is a
**contextvar**: ``collect_latency()`` installs a collector for the duration of a
``with`` block, and any ``measure("embedding")`` running underneath - however deep -
records into it. Outside a collector, ``measure`` is just a timer whose result is
dropped. This is the same pattern tracing libraries use.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel

# canonical stage order for reports
STAGES = ("embedding", "retrieval", "reranking", "generation", "evaluation", "total")

_active_collector: ContextVar[Optional["LatencyCollector"]] = ContextVar(
    "latency_collector", default=None
)


class LatencyCollector:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def add(self, stage: str, ms: float) -> None:
        # accumulate: a stage measured twice in one execution sums
        self.timings[stage] = round(self.timings.get(stage, 0.0) + ms, 3)


@contextmanager
def collect_latency() -> Iterator[LatencyCollector]:
    """Install a collector; ``measure(...)`` calls inside the block feed it."""
    collector = LatencyCollector()
    token = _active_collector.set(collector)
    try:
        yield collector
    finally:
        _active_collector.reset(token)


@contextmanager
def measure(stage: str) -> Iterator[None]:
    """Time this block. If a collector is active, add the duration (ms) to it."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        collector = _active_collector.get()
        if collector is not None:
            collector.add(stage, elapsed_ms)


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default)."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (p / 100.0) * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


class LatencyStats(BaseModel):
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "LatencyStats":
        ordered = sorted(values)
        return cls(
            count=len(ordered),
            mean_ms=round(statistics.fmean(ordered), 3),
            median_ms=round(statistics.median(ordered), 3),
            p95_ms=round(_percentile(ordered, 95.0), 3),
            min_ms=round(ordered[0], 3),
            max_ms=round(ordered[-1], 3),
        )


class LatencyReport(BaseModel):
    """Per-stage latency distribution across every question in a run."""

    stages: dict[str, LatencyStats]

    def mean(self, stage: str) -> float:
        s = self.stages.get(stage)
        return s.mean_ms if s else 0.0

    def p95(self, stage: str) -> float:
        s = self.stages.get(stage)
        return s.p95_ms if s else 0.0

    @classmethod
    def from_question_timings(
        cls, per_question: Sequence[dict]
    ) -> "LatencyReport":
        by_stage: dict[str, list[float]] = {}
        for timings in per_question:
            for stage, ms in timings.items():
                by_stage.setdefault(stage, []).append(ms)

        ordered_names = [s for s in STAGES if s in by_stage]
        ordered_names += [s for s in by_stage if s not in STAGES]
        return cls(
            stages={name: LatencyStats.from_values(by_stage[name]) for name in ordered_names}
        )
