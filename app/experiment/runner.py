"""run_experiment(config) - the driver that ties Phases 1-6 together.

For each question: retrieve -> generate (plain or cited) -> run the enabled
evaluators, timing every stage, metering evaluation tokens, and catching per-question
errors so one bad question doesn't sink the run. Then aggregate and assemble an
``ExperimentResult`` (reusing the Phase 3-6 ``aggregate_*`` functions) and save it.

``ExperimentComponents`` bundles everything the loop needs. In production the factory
builds it from the config; tests pass a fake bundle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from app.config import settings
from app.evaluation.citation import CitationEvaluator, LLMCitationEvaluator
from app.evaluation.citation_eval import (
    aggregate_citation_metrics,
    evaluate_citations_for_question,
)
from app.evaluation.dataset import EvalExample, load_eval_dataset
from app.evaluation.faithfulness import FaithfulnessEvaluator, LLMFaithfulnessEvaluator
from app.evaluation.faithfulness_eval import (
    aggregate_faithfulness_metrics,
    evaluate_faithfulness_for_question,
)
from app.evaluation.generation import (
    aggregate_generation_metrics,
    evaluate_generation_for_question,
)
from app.evaluation.judge import GenerationJudge, LLMGenerationJudge
from app.evaluation.retrieval import (
    aggregate_retrieval_metrics,
    evaluate_retrieval_for_question,
)
from app.experiment.config import ExperimentConfig
from app.experiment.cost import (
    apply_configured_pricing,
    embedding_cost,
    llm_cost,
)
from app.experiment.metering import RecordingTextLLM, TokenMeter, add_usage
from app.experiment.results import (
    CostBreakdown,
    ExperimentError,
    ExperimentResult,
    LatencySummary,
    QuestionCost,
    QuestionExperimentResult,
)
from app.observability.latency import LatencyReport, collect_latency, measure
from app.observability.recorder import TraceRecorder
from app.observability.trace import PerformanceTrace, Trace
from app.generation.citation import AnthropicCitedGenerator, CitedGenerator
from app.generation.generator import AnthropicGenerator, LLMGenerator
from app.ingestion.chunker import TextChunker
from app.ingestion.embeddings import SentenceTransformerEmbeddingProvider
from app.llm import AnthropicTextLLM
from app.models import TokenUsage
from app.pipeline import RagPipeline
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore


@dataclass
class ExperimentComponents:
    pipeline: RagPipeline
    generation_model: str
    judge_model: str
    embedding_model: str
    eval_meter: TokenMeter
    cited_generator: Optional[CitedGenerator] = None
    judge: Optional[GenerationJudge] = None
    faithfulness_evaluator: Optional[FaithfulnessEvaluator] = None
    citation_evaluator: Optional[CitationEvaluator] = None


def build_experiment_components(
    config: ExperimentConfig, *, api_key: Optional[str] = None
) -> ExperimentComponents:
    if config.reranker_enabled:
        raise NotImplementedError(
            "reranker_enabled is not implemented yet (extension point for a later phase)"
        )
    if config.retrieval_strategy != "dense":
        raise NotImplementedError(
            f"retrieval_strategy={config.retrieval_strategy!r} not implemented; only 'dense'"
        )

    embeddings = SentenceTransformerEmbeddingProvider(config.embedding_model)
    store = InMemoryVectorStore()

    # generation LLM is NOT metered (tokens come from the answer object);
    # evaluation LLM IS metered.
    gen_llm = AnthropicTextLLM(model=config.generation_model, api_key=api_key)
    meter = TokenMeter()
    eval_llm = RecordingTextLLM(
        AnthropicTextLLM(model=config.effective_judge_model, api_key=api_key), meter
    )

    pipeline = RagPipeline(
        chunker=TextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap),
        embeddings=embeddings,
        vector_store=store,
        retriever=DenseRetriever(embeddings, store),
        generator=AnthropicGenerator(llm=gen_llm, max_tokens=settings.generation_max_tokens),
        default_top_k=config.top_k,
    )

    return ExperimentComponents(
        pipeline=pipeline,
        generation_model=config.generation_model,
        judge_model=config.effective_judge_model,
        embedding_model=config.embedding_model,
        eval_meter=meter,
        cited_generator=(
            AnthropicCitedGenerator(llm=gen_llm, max_tokens=settings.generation_max_tokens)
            if config.citations_enabled
            else None
        ),
        judge=(
            LLMGenerationJudge(eval_llm, max_retries=config.max_retries)
            if (config.run_generation_eval and config.use_judge)
            else None
        ),
        faithfulness_evaluator=(
            LLMFaithfulnessEvaluator(eval_llm, max_retries=config.max_retries)
            if config.run_faithfulness
            else None
        ),
        citation_evaluator=(
            LLMCitationEvaluator(eval_llm, max_retries=config.max_retries)
            if config.citation_eval_enabled
            else None
        ),
    )


def _generate(question: str, retrieved, config: ExperimentConfig, comps: ExperimentComponents):
    """Return the full GeneratedAnswer / CitedAnswer from the plain or cited generator."""
    if config.citations_enabled:
        return comps.cited_generator.generate(question, list(retrieved))
    gen: LLMGenerator = comps.pipeline.generator
    return gen.generate(question, list(retrieved))


def _run_one(
    example: EvalExample, config: ExperimentConfig, comps: ExperimentComponents
) -> tuple[QuestionExperimentResult, Optional[Trace]]:
    meter_before = comps.eval_meter.snapshot()

    with collect_latency() as latency_collector, measure("total"):
        # DenseRetriever records "embedding" and "retrieval" into the active collector
        retrieved = comps.pipeline.retrieve(example.question, config.top_k)

        with measure("reranking"):
            pass  # extension point: no reranker yet, but the stage is always timed

        with measure("generation"):
            answer = _generate(example.question, retrieved, config, comps)
            answer_text = answer.answer
            gen_usage = answer.token_usage

        with measure("evaluation"):
            retrieval_res = (
                evaluate_retrieval_for_question(retrieved, example, config.top_k)
                if config.run_retrieval_eval
                else None
            )
            generation_res = (
                evaluate_generation_for_question(
                    example,
                    answer_text,
                    judge=comps.judge if config.use_judge else None,
                )
                if config.run_generation_eval
                else None
            )
            faithfulness_res = (
                evaluate_faithfulness_for_question(
                    example.id, answer_text, retrieved, comps.faithfulness_evaluator
                )
                if config.run_faithfulness
                else None
            )
            citation_res = (
                evaluate_citations_for_question(
                    example.id, answer_text, retrieved, comps.citation_evaluator
                )
                if config.citation_eval_enabled
                else None
            )

    latency = latency_collector.timings
    eval_usage = comps.eval_meter.delta_since(meter_before)
    query_embedding_tokens = comps.pipeline.embeddings.count_tokens(example.question)
    question_usage = add_usage(
        add_usage(eval_usage, gen_usage), TokenUsage(embedding_tokens=query_embedding_tokens)
    )

    generation_usd = llm_cost(gen_usage or TokenUsage(), comps.generation_model)
    evaluation_usd = llm_cost(eval_usage, comps.judge_model)
    query_embedding_usd = embedding_cost(query_embedding_tokens, comps.embedding_model)
    question_cost = QuestionCost(
        query_embedding_usd=query_embedding_usd,
        generation_usd=generation_usd,
        evaluation_usd=evaluation_usd,
        total_usd=query_embedding_usd + generation_usd + evaluation_usd,
    )

    result = QuestionExperimentResult(
        question_id=example.id,
        question=example.question,
        slices=list(example.slices),
        retrieved_chunk_ids=[rc.chunk.chunk_id for rc in retrieved],
        retrieved_doc_ids=sorted({rc.chunk.document_id for rc in retrieved}),
        generated_answer=answer_text,
        retrieval=retrieval_res,
        generation=generation_res,
        faithfulness=faithfulness_res,
        citation=citation_res,
        latency_ms=latency,
        token_usage=question_usage,
        estimated_cost_usd=question_cost.total_usd,
        cost=question_cost,
    )

    trace = None
    if config.tracing_enabled:
        recorder = TraceRecorder(example.question, example.id, latency=latency)
        recorder.record_retrieval(
            query=example.question,
            retrieved=retrieved,
            top_k=config.top_k,
            embedding_model=config.embedding_model,
            embedding_dim=getattr(comps.pipeline.embeddings, "dimension", None),
        )
        recorder.record_generation(
            model=answer.model,
            prompt=answer.prompt or "",
            answer=answer_text,
            token_usage=answer.token_usage,
            citations=getattr(answer, "citations", None),
        )
        recorder.record_evaluation(
            retrieval_metrics=retrieval_res.metrics if retrieval_res else None,
            judgement=generation_res.judgement if generation_res else None,
            deterministic=generation_res.deterministic if generation_res else None,
            faithfulness=faithfulness_res.result if faithfulness_res else None,
            citation=citation_res.result if citation_res else None,
        )
        trace = recorder.build(
            token_usage=question_usage, estimated_cost_usd=question_cost.total_usd
        )

    return result, trace


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "experiment"


def run_experiment(
    config: ExperimentConfig,
    *,
    api_key: Optional[str] = None,
    components: Optional[ExperimentComponents] = None,
    dataset: Optional[Iterable[EvalExample]] = None,
) -> ExperimentResult:
    started = datetime.now(timezone.utc)
    apply_configured_pricing()
    comps = components or build_experiment_components(config, api_key=api_key)

    ingestion = comps.pipeline.ingest(config.documents_dir or settings.documents_dir)

    if dataset is not None:
        examples = list(dataset)
    else:
        examples = list(load_eval_dataset(config.dataset_path))
    if config.limit:
        examples = examples[: config.limit]

    per_question: list[QuestionExperimentResult] = []
    traces: list[Trace] = []
    errors: list[ExperimentError] = []
    for example in examples:
        try:
            question_result, trace = _run_one(example, config, comps)
            per_question.append(question_result)
            if trace is not None:
                traces.append(trace)
        except Exception as exc:  # noqa: BLE001 - one bad question must not sink the run
            errors.append(
                ExperimentError(question_id=example.id, stage="run", message=repr(exc))
            )
            per_question.append(
                QuestionExperimentResult(
                    question_id=example.id,
                    question=example.question,
                    slices=list(example.slices),
                    error=repr(exc),
                )
            )
            if config.tracing_enabled:
                traces.append(
                    Trace(
                        trace_id=uuid4().hex,
                        question=example.question,
                        question_id=example.id,
                        started_at=datetime.now(timezone.utc).isoformat(),
                        performance=PerformanceTrace(),
                        errors=[repr(exc)],
                    )
                )

    ok = [q for q in per_question if q.ok]

    retrieval_results = [q.retrieval for q in ok if q.retrieval is not None]
    generation_results = [q.generation for q in ok if q.generation is not None]
    faithfulness_results = [q.faithfulness for q in ok if q.faithfulness is not None]
    citation_results = [q.citation for q in ok if q.citation is not None]

    total_usage = TokenUsage()
    for q in ok:
        total_usage = add_usage(total_usage, q.token_usage)

    ingestion_embedding_usd = embedding_cost(ingestion.embedding_tokens, config.embedding_model)
    query_embedding_usd = sum(q.cost.query_embedding_usd for q in ok)
    generation_usd = sum(q.cost.generation_usd for q in ok)
    evaluation_usd = sum(q.cost.evaluation_usd for q in ok)
    marginal = query_embedding_usd + generation_usd + evaluation_usd
    cost = CostBreakdown(
        ingestion_embedding_usd=round(ingestion_embedding_usd, 8),
        query_embedding_usd=round(query_embedding_usd, 8),
        generation_usd=round(generation_usd, 6),
        evaluation_usd=round(evaluation_usd, 6),
        total_usd=round(ingestion_embedding_usd + marginal, 6),
        cost_per_query_usd=round(marginal / len(ok), 6) if ok else 0.0,
    )

    finished = datetime.now(timezone.utc)
    return ExperimentResult(
        experiment_id=f"{_slugify(config.experiment_name)}_{started.strftime('%Y%m%d-%H%M%S')}",
        config=config,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        num_questions=len(per_question),
        num_errors=len(errors),
        document_count=ingestion.document_count,
        chunk_count=ingestion.chunk_count,
        per_question=per_question,
        traces=traces,
        retrieval=aggregate_retrieval_metrics(retrieval_results) if retrieval_results else None,
        generation=aggregate_generation_metrics(generation_results) if generation_results else None,
        faithfulness=(
            aggregate_faithfulness_metrics(faithfulness_results) if faithfulness_results else None
        ),
        citation=aggregate_citation_metrics(citation_results) if citation_results else None,
        latency=LatencySummary(
            embedding_ms=_mean(q.latency_ms.get("embedding", 0.0) for q in ok),
            retrieval_ms=_mean(q.latency_ms.get("retrieval", 0.0) for q in ok),
            reranking_ms=_mean(q.latency_ms.get("reranking", 0.0) for q in ok),
            generation_ms=_mean(q.latency_ms.get("generation", 0.0) for q in ok),
            evaluation_ms=_mean(q.latency_ms.get("evaluation", 0.0) for q in ok),
            total_ms=_mean(q.latency_ms.get("total", 0.0) for q in ok),
        ),
        latency_report=(
            LatencyReport.from_question_timings([q.latency_ms for q in ok]) if ok else None
        ),
        total_token_usage=total_usage,
        estimated_cost_usd=cost.total_usd,
        cost=cost,
        errors=errors,
    )


def save_experiment(result: ExperimentResult, directory: Optional[str] = None) -> Path:
    directory = Path(directory or settings.experiments_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{result.experiment_id}.json"
    path.write_text(json.dumps(result.model_dump(), indent=2) + "\n", encoding="utf-8")
    return path


def load_experiment(path: str | Path) -> ExperimentResult:
    return ExperimentResult.model_validate_json(Path(path).read_text(encoding="utf-8"))
