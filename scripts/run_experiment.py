"""Phase 7 CLI: run one configurable experiment end to end.

    # from a config file
    python -m scripts.run_experiment --config data/experiments/chunk500.json

    # or from flags
    python -m scripts.run_experiment --name chunk300_top3 --chunk-size 300 --top-k 3 \
        --faithfulness --limit 5

COST: 1 generation call per question, plus 1 judge call (unless --no-judge), plus
1 faithfulness call (with --faithfulness), plus 1 citation call (with --citations).
Use --limit while iterating.
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.experiment.config import ExperimentConfig, load_experiment_config
from app.experiment.results import ExperimentResult
from app.experiment.runner import run_experiment, save_experiment
from app.experiment.store import ExperimentStore

RULE = "=" * 78


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config:
        return load_experiment_config(args.config)
    if not args.name:
        raise SystemExit("provide --config PATH or --name NAME")
    return ExperimentConfig(
        experiment_name=args.name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        generation_model=args.model,
        citations_enabled=args.citations,
        use_judge=not args.no_judge,
        run_faithfulness=args.faithfulness,
        limit=args.limit,
    )


def print_report(result: ExperimentResult) -> None:
    print(f"\n{RULE}\nEXPERIMENT  {result.experiment_id}\n{RULE}")
    c = result.config
    print(f"config: chunk={c.chunk_size}/{c.chunk_overlap} top_k={c.top_k} "
          f"model={c.generation_model} citations={c.citations_enabled}")
    print(f"questions: {result.num_questions}  errors: {result.num_errors}  "
          f"corpus: {result.document_count} docs / {result.chunk_count} chunks")

    if result.retrieval:
        r = result.retrieval
        print(f"\nretrieval @k={r.k}: hit={r.hit_rate:.3f} precision={r.precision:.3f} "
              f"recall={r.recall:.3f} mrr={r.mrr:.3f} ndcg={r.ndcg:.3f}")
    if result.generation:
        g = result.generation
        print(f"generation: exact={g.exact_match:.3f} f1={g.token_f1:.3f} "
              f"abstention_acc={g.abstention_accuracy:.3f}", end="")
        if g.num_judged:
            print(f" | judge correctness={g.judge_correctness:.3f} relevance={g.judge_relevance:.3f}")
        else:
            print()
    if result.faithfulness:
        f = result.faithfulness
        print(f"faithfulness: macro={f.faithfulness:.3f} micro={f.claim_support_rate:.3f}")
    if result.citation:
        ct = result.citation
        print(f"citations: completeness={ct.citation_completeness:.3f} "
              f"precision={ct.citation_precision:.3f} halluc={ct.citation_hallucination_rate:.3f}")

    print("\nlatency ms (mean / p95):")
    if result.latency_report:
        for stage, st in result.latency_report.stages.items():
            print(f"  {stage:<12} {st.mean_ms:>8.1f} / {st.p95_ms:.1f}")
    else:
        lat = result.latency
        print(f"  retrieval={lat.retrieval_ms:.0f} generation={lat.generation_ms:.0f} "
              f"evaluation={lat.evaluation_ms:.0f} total={lat.total_ms:.0f}")
    tok = result.total_token_usage
    print(f"tokens: embedding={tok.embedding_tokens} prompt={tok.prompt_tokens} "
          f"completion={tok.completion_tokens} total={tok.total_tokens}")

    c = result.cost
    print("cost (USD):")
    print(f"  ingestion embedding  ${c.ingestion_embedding_usd:.6f}  (one-time)")
    print(f"  query embedding      ${c.query_embedding_usd:.6f}")
    print(f"  generation           ${c.generation_usd:.6f}")
    print(f"  evaluation           ${c.evaluation_usd:.6f}")
    print(f"  total                ${c.total_usd:.6f}   ({c.cost_per_query_usd:.6f}/query marginal)")

    qname, qval = result.headline_quality()
    if qval is not None and c.cost_per_query_usd > 0:
        print(f"quality vs cost: {qname}={qval:.3f} @ ${c.cost_per_query_usd:.6f}/query")

    for err in result.errors:
        print(f"  ERROR {err.question_id}: {err.message}")


def main() -> None:
    p = argparse.ArgumentParser(description="Run one configurable RAG experiment.")
    p.add_argument("--config", help="Path to an ExperimentConfig JSON file.")
    p.add_argument("--name")
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    p.add_argument("--top-k", type=int, default=settings.top_k)
    p.add_argument("--model", default=settings.generation_model)
    p.add_argument("--citations", action="store_true")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--faithfulness", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-save", action="store_true", help="Do not persist the result.")
    p.add_argument("--json", action="store_true", help="Also write a result JSON file.")
    args = p.parse_args()

    config = config_from_args(args)
    print(f"Running experiment '{config.experiment_name}'...")
    result = run_experiment(config, api_key=settings.anthropic_api_key)
    print_report(result)

    if not args.no_save:
        with ExperimentStore() as store:
            store.save(result)
        print(f"\nsaved to db -> {settings.experiments_db}  (id: {result.experiment_id})")
        if args.json:
            print(f"saved json  -> {save_experiment(result)}")


if __name__ == "__main__":
    main()
