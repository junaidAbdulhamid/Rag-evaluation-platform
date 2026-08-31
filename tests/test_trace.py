"""Tests for the TraceRecorder / Trace model."""

from __future__ import annotations

import time

from app.evaluation.faithfulness import Claim, FaithfulnessResult
from app.evaluation.generation import DeterministicScores
from app.evaluation.judge import GenerationJudgement
from app.evaluation.retrieval import RetrievalMetrics
from app.models import Chunk, Citation, RetrievedChunk, TokenUsage
from app.observability.recorder import TraceRecorder


def chunk(doc_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"{doc_id}::c{rank}", document_id=doc_id, text="x " * 300),
        score=1.0 / rank,
        rank=rank,
    )


def test_recorder_builds_a_full_trace():
    rec = TraceRecorder("How long for a refund?", "q1")

    with rec.stage("retrieval"):
        time.sleep(0.002)
    rec.record_retrieval(
        query="How long for a refund?",
        retrieved=[chunk("refund_policy", 1), chunk("warranty_policy", 2)],
        top_k=2,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
    )
    rec.record_generation(
        model="claude-opus-5",
        prompt="Context: ...\nQuestion: ...",
        answer="Customers have 30 days. [1]",
        token_usage=TokenUsage(prompt_tokens=900, completion_tokens=20, total_tokens=920),
        citations=[Citation(marker=1, exists=True, chunk_id="refund_policy::c1", document_id="refund_policy")],
    )
    rec.record_evaluation(
        retrieval_metrics=RetrievalMetrics(
            k=2, hit_rate=1.0, precision=0.5, recall=1.0, reciprocal_rank=1.0,
            ndcg=1.0, num_relevant=1, num_retrieved_docs=2,
        ),
        judgement=GenerationJudgement(
            correctness=0.9, relevance=1.0, correctness_reasoning="ok", relevance_reasoning="ok"
        ),
        deterministic=DeterministicScores(
            exact_match=0.0, token_f1=0.8, token_recall=1.0, number_coverage=1.0,
            abstention_expected=False, abstained=False, abstention_correct=True,
        ),
        faithfulness=FaithfulnessResult.from_claims([Claim(text="c", supported=True, reason="r")]),
    )

    trace = rec.build(
        token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=25, total_tokens=1025),
        estimated_cost_usd=0.002,
    )

    assert trace.question_id == "q1" and trace.ok
    # retrieval section
    assert trace.retrieval.top_k == 2
    assert [c.rank for c in trace.retrieval.chunks] == [1, 2]
    assert trace.retrieval.chunks[0].chunk_id == "refund_policy::c1"
    assert trace.retrieval.chunks[0].text_preview.endswith("...")  # long text truncated
    assert trace.retrieval.latency_ms >= 1.0  # captured from the stage() block
    # generation section
    assert trace.generation.model == "claude-opus-5"
    assert trace.generation.citations[0].marker == 1
    # evaluation section - judgement is flattened to scalars
    assert trace.evaluation.correctness == 0.9
    assert trace.evaluation.relevance == 1.0
    assert trace.evaluation.faithfulness.score == 1.0
    # performance section
    assert trace.performance.token_usage.total_tokens == 1025
    assert trace.performance.estimated_cost_usd == 0.002


def test_recorder_uses_a_shared_latency_dict():
    shared: dict = {"retrieval": 12.0, "generation": 340.0, "evaluation": 900.0, "total": 1252.0}
    rec = TraceRecorder("q?", "q1", latency=shared)
    rec.record_retrieval(query="q?", retrieved=[chunk("d", 1)], top_k=1)

    trace = rec.build(token_usage=TokenUsage(), estimated_cost_usd=0.0)

    assert trace.retrieval.latency_ms == 12.0
    assert trace.performance.generation_ms == 340.0
    assert trace.performance.total_ms == 1252.0


def test_recorder_records_errors():
    rec = TraceRecorder("q?", "q1")
    rec.record_error("RuntimeError('boom')")
    trace = rec.build(token_usage=TokenUsage(), estimated_cost_usd=0.0)

    assert not trace.ok
    assert trace.errors == ["RuntimeError('boom')"]
