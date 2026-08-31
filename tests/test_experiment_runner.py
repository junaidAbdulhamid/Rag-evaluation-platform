"""End-to-end tests for run_experiment, driven entirely by fakes (no API, no torch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.citation import CitationEvaluationResult, CitedClaim, CitationLink
from app.evaluation.dataset import EvalExample
from app.evaluation.faithfulness import Claim, FaithfulnessResult
from app.evaluation.judge import GenerationJudge, GenerationJudgement, LLMGenerationJudge
from app.experiment.config import ExperimentConfig
from app.experiment.metering import RecordingTextLLM, TokenMeter
from app.experiment.runner import (
    ExperimentComponents,
    load_experiment,
    run_experiment,
    save_experiment,
)
from app.ingestion.chunker import TextChunker
from app.pipeline import RagPipeline
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from tests.fakes import (
    FakeCitationEvaluator,
    FakeEmbeddingProvider,
    FakeFaithfulnessEvaluator,
    FakeTextLLM,
    FixedAnswerGenerator,
)

ANSWER = "Customers have 30 days to request a refund."
JUDGE_JSON = json.dumps(
    {"correctness": 0.9, "relevance": 1.0, "correctness_reasoning": "ok", "relevance_reasoning": "ok"}
)


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "refund_policy.md").write_text(
        "Customers have 30 days from the delivery date to request a refund.", encoding="utf-8"
    )
    (tmp_path / "shipping_policy.md").write_text(
        "Standard shipping takes three to five business days.", encoding="utf-8"
    )
    return tmp_path


def dataset() -> list[EvalExample]:
    return [
        EvalExample(id="q1", question="How long for a refund?", expected_answer=ANSWER,
                    relevant_document_ids=["refund_policy"]),
        EvalExample(id="q2", question="How fast is shipping?",
                    expected_answer="Three to five business days.",
                    relevant_document_ids=["shipping_policy"]),
    ]


def build_components(judge: GenerationJudge | None = None) -> tuple[ExperimentComponents, TokenMeter]:
    embeddings = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    meter = TokenMeter()
    pipeline = RagPipeline(
        chunker=TextChunker(chunk_size=200, chunk_overlap=20),
        embeddings=embeddings,
        vector_store=store,
        retriever=DenseRetriever(embeddings, store),
        generator=FixedAnswerGenerator(ANSWER),
        default_top_k=2,
    )
    faith = FaithfulnessResult.from_claims([Claim(text="c", supported=True, reason="r")])
    cite = CitationEvaluationResult.compute(
        [CitedClaim(text="c", markers=[1], has_citation=True)],
        [CitationLink(claim_index=0, claim_text="c", marker=1, exists=True,
                      resolved_chunk_id="refund_policy::chunk_0", supports_claim=True)],
    )
    comps = ExperimentComponents(
        pipeline=pipeline,
        generation_model="claude-opus-5",
        judge_model="claude-opus-5",
        eval_meter=meter,
        judge=judge or LLMGenerationJudge(RecordingTextLLM(FakeTextLLM(JUDGE_JSON), meter)),
        faithfulness_evaluator=FakeFaithfulnessEvaluator({ANSWER: faith}),
        citation_evaluator=FakeCitationEvaluator({ANSWER: cite}),
    )
    return comps, meter


def base_config(**overrides) -> ExperimentConfig:
    kwargs = dict(experiment_name="test", top_k=2, run_faithfulness=True, run_citation_eval=True)
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_full_run_populates_every_section(corpus_dir: Path):
    comps, meter = build_components()
    result = run_experiment(
        base_config(documents_dir=str(corpus_dir)), components=comps, dataset=dataset()
    )

    assert result.num_questions == 2 and result.num_errors == 0
    assert result.document_count == 2

    q = result.per_question[0]
    assert q.retrieved_chunk_ids and q.generated_answer == ANSWER
    assert q.retrieval is not None and q.generation is not None
    assert q.faithfulness is not None and q.citation is not None
    assert q.latency_ms["total"] >= 0.0

    # aggregates reuse the Phase 3-6 functions
    assert result.retrieval is not None and result.retrieval.k == 2
    assert result.generation is not None and result.generation.num_judged == 2
    assert result.faithfulness is not None
    assert result.citation is not None

    # tokens: 15/question from the generator + 2/question from the (metered) judge
    assert q.token_usage.total_tokens == 17
    assert result.total_token_usage.total_tokens == 34
    assert result.estimated_cost_usd > 0.0


def test_evaluator_toggles_are_respected(corpus_dir: Path):
    comps, _ = build_components()
    result = run_experiment(
        base_config(documents_dir=str(corpus_dir), run_faithfulness=False, run_citation_eval=False,
                    use_judge=False),
        components=comps,
        dataset=dataset(),
    )
    q = result.per_question[0]
    assert q.faithfulness is None and q.citation is None
    assert q.generation is not None and q.generation.judgement is None  # deterministic only
    assert result.faithfulness is None and result.citation is None


def test_per_question_error_is_isolated(corpus_dir: Path):
    class BoomJudge(GenerationJudge):
        def judge(self, *, question, expected_answer, generated_answer):
            if "shipping" in question:
                raise RuntimeError("judge exploded")
            return GenerationJudgement(correctness=1.0, relevance=1.0,
                                       correctness_reasoning="a", relevance_reasoning="b")

    comps, _ = build_components(judge=BoomJudge())
    result = run_experiment(
        base_config(documents_dir=str(corpus_dir)), components=comps, dataset=dataset()
    )

    assert result.num_questions == 2
    assert result.num_errors == 1
    assert result.errors[0].question_id == "q2"
    assert result.per_question[1].error is not None and not result.per_question[1].ok
    # q1 still fully evaluated; aggregates computed over the survivor
    assert result.per_question[0].ok
    assert result.generation is not None and result.generation.num_questions == 1


def test_save_and_load_round_trip(corpus_dir: Path):
    comps, _ = build_components()
    result = run_experiment(
        base_config(documents_dir=str(corpus_dir)), components=comps, dataset=dataset()
    )
    path = save_experiment(result, directory=str(corpus_dir / "experiments"))
    assert path.exists()

    loaded = load_experiment(path)
    assert loaded.experiment_id == result.experiment_id
    assert loaded.config == result.config
    assert loaded.per_question[0].generated_answer == ANSWER
