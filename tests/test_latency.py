"""Tests for fine-grained latency: contextvar collection + distribution stats."""

from __future__ import annotations

import time

import pytest

from app.observability.latency import (
    LatencyReport,
    LatencyStats,
    _percentile,
    collect_latency,
    measure,
)


# --- collect_latency / measure ---------------------------------------------------------
def test_measure_outside_a_collector_is_a_harmless_noop():
    with measure("retrieval"):  # nothing installed - must not raise
        pass


def test_collector_records_nested_measures():
    with collect_latency() as collector:
        with measure("embedding"):
            time.sleep(0.003)
        with measure("retrieval"):
            time.sleep(0.001)

    assert set(collector.timings) == {"embedding", "retrieval"}
    assert collector.timings["embedding"] >= 2.0
    assert collector.timings["embedding"] > collector.timings["retrieval"]


def test_same_stage_measured_twice_accumulates():
    with collect_latency() as collector:
        with measure("evaluation"):
            time.sleep(0.002)
        with measure("evaluation"):
            time.sleep(0.002)
    assert collector.timings["evaluation"] >= 3.5  # ~4ms total


def test_collectors_do_not_leak_between_blocks():
    with collect_latency() as first:
        with measure("a"):
            pass
    with collect_latency() as second:
        with measure("b"):
            pass
    assert "a" in first.timings and "a" not in second.timings


# --- percentile / LatencyStats -----------------------------------------------------------
def test_percentile_interpolates():
    values = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert _percentile(values, 0) == 0.0
    assert _percentile(values, 50) == 20.0
    assert _percentile(values, 95) == pytest.approx(38.0)  # between 30 and 40


def test_latency_stats_from_values():
    st = LatencyStats.from_values([10.0, 20.0, 30.0, 40.0, 1000.0])
    assert st.count == 5
    assert st.mean_ms == 220.0
    assert st.median_ms == 30.0
    assert st.min_ms == 10.0 and st.max_ms == 1000.0
    assert st.p95_ms > st.median_ms


def test_latency_stats_single_value():
    st = LatencyStats.from_values([42.0])
    assert (st.mean_ms, st.median_ms, st.p95_ms, st.min_ms, st.max_ms) == (42.0, 42.0, 42.0, 42.0, 42.0)


# --- LatencyReport ---------------------------------------------------------------------
def test_report_groups_by_stage_in_canonical_order():
    per_q = [
        {"total": 100.0, "embedding": 5.0, "retrieval": 3.0, "generation": 40.0, "evaluation": 50.0},
        {"total": 200.0, "embedding": 6.0, "retrieval": 4.0, "generation": 90.0, "evaluation": 95.0},
    ]
    report = LatencyReport.from_question_timings(per_q)

    assert list(report.stages) == ["embedding", "retrieval", "generation", "evaluation", "total"]
    assert report.stages["total"].mean_ms == 150.0
    assert report.mean("generation") == 65.0
    assert report.p95("total") > 150.0
    assert report.mean("reranking") == 0.0  # absent stage -> 0
