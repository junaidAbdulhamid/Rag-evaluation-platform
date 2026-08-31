"""Phase 8 CLI: manage the tracked experiments database.

    python -m scripts.experiments list
    python -m scripts.experiments show    <experiment_id>
    python -m scripts.experiments metrics <experiment_id>
    python -m scripts.experiments delete  <experiment_id>
"""

from __future__ import annotations

import argparse

from app.experiment.store import ExperimentStore

RULE = "=" * 78


def cmd_list(store: ExperimentStore, args: argparse.Namespace) -> None:
    rows = store.list(limit=args.limit)
    if not rows:
        print("no experiments tracked yet.")
        return
    print(f"{'experiment_id':<34} {'model':<20} chunk  k  n   recall  correct  cost")
    print("-" * 96)
    for s in rows:

        def f(v, spec="{:.3f}"):
            return spec.format(v) if v is not None else "  -  "

        print(f"{s.experiment_id:<34} {s.generation_model:<20} "
              f"{s.chunk_size:<5} {s.top_k:<2} {s.num_questions:<3} "
              f"{f(s.retrieval_recall)}   {f(s.judge_correctness)}   ${s.estimated_cost_usd:.4f}")


def cmd_show(store: ExperimentStore, args: argparse.Namespace) -> None:
    result = store.get(args.experiment_id)
    if result is None:
        raise SystemExit(f"no such experiment: {args.experiment_id}")
    print(f"{RULE}\n{result.experiment_id}\n{RULE}")
    print(f"config     : {result.config.model_dump()}")
    print(f"started    : {result.started_at}")
    print(f"questions  : {result.num_questions}  errors: {result.num_errors}")
    print(f"corpus     : {result.document_count} docs / {result.chunk_count} chunks")
    print(f"tokens     : {result.total_token_usage.total_tokens}  cost: ${result.estimated_cost_usd:.4f}")
    print("\nper-question:")
    for q in result.per_question:
        rec = q.retrieval.metrics.recall if q.retrieval else None
        cor = q.generation.judgement.correctness if (q.generation and q.generation.judgement) else None
        flag = "  ERROR" if q.error else ""
        print(f"  {q.question_id:<6} recall={rec}  judge_correct={cor}{flag}")


def cmd_metrics(store: ExperimentStore, args: argparse.Namespace) -> None:
    result = store.get(args.experiment_id)
    if result is None:
        raise SystemExit(f"no such experiment: {args.experiment_id}")
    print(f"{RULE}\nMETRICS  {result.experiment_id}\n{RULE}")
    for name, agg in (
        ("retrieval", result.retrieval),
        ("generation", result.generation),
        ("faithfulness", result.faithfulness),
        ("citation", result.citation),
    ):
        print(f"\n[{name}]")
        if agg is None:
            print("  (not run)")
        else:
            for k, v in agg.model_dump().items():
                print(f"  {k:<26} {v}")
    lat = result.latency
    print(f"\n[latency mean ms]\n  retrieval={lat.retrieval_ms:.0f} generation={lat.generation_ms:.0f} "
          f"evaluation={lat.evaluation_ms:.0f} total={lat.total_ms:.0f}")


def cmd_delete(store: ExperimentStore, args: argparse.Namespace) -> None:
    if store.delete(args.experiment_id):
        print(f"deleted {args.experiment_id}")
    else:
        raise SystemExit(f"no such experiment: {args.experiment_id}")


def cmd_trace(store: ExperimentStore, args: argparse.Namespace) -> None:
    traces = store.get_traces(args.experiment_id)
    if not traces:
        raise SystemExit(f"no traces for experiment: {args.experiment_id}")
    if args.question:
        traces = [t for t in traces if t.question_id == args.question]
        if not traces:
            raise SystemExit(f"no trace for question {args.question}")
    for trace in traces if args.all else traces[:1]:
        _print_trace(trace)


def _print_trace(t) -> None:
    print(f"\n{RULE}\nTrace {t.trace_id}   ({t.question_id})\n{RULE}")
    print(f"|-- question: {t.question}")
    if t.retrieval:
        r = t.retrieval
        print(f"|-- retrieval  [{r.latency_ms:.1f} ms]   "
              f"embedding={r.embedding_model} dim={r.embedding_dim} top_k={r.top_k}")
        for c in r.chunks:
            print(f"|   #{c.rank} {c.score:.4f} {c.document_id}::{c.chunk_id.split('::')[-1]}  "
                  f"\"{c.text_preview[:80]}\"")
    if t.generation:
        g = t.generation
        toks = f"prompt={g.token_usage.prompt_tokens} completion={g.token_usage.completion_tokens}" if g.token_usage else "n/a"
        print(f"|-- generation  [{g.latency_ms:.1f} ms]   model={g.model}  tokens: {toks}")
        print(f"|   prompt ({len(g.prompt)} chars): \"{g.prompt[:120].replace(chr(10), ' ')}...\"")
        print(f"|   answer: \"{g.answer[:160]}\"")
        if g.citations:
            print(f"|   citations: {[(c.marker, c.chunk_id) for c in g.citations]}")
    if t.evaluation:
        e = t.evaluation
        print(f"|-- evaluation  [{e.latency_ms:.1f} ms]")
        if e.retrieval_metrics:
            m = e.retrieval_metrics
            print(f"|   retrieval: hit={m.hit_rate} recall={m.recall} mrr={m.reciprocal_rank} ndcg={m.ndcg}")
        print(f"|   correctness={e.correctness}  relevance={e.relevance}")
        if e.faithfulness:
            print(f"|   faithfulness={e.faithfulness.score} "
                  f"({e.faithfulness.num_supported}/{e.faithfulness.num_claims} claims)")
        if e.citation:
            print(f"|   citations: precision={e.citation.citation_precision} "
                  f"completeness={e.citation.citation_completeness} "
                  f"halluc={e.citation.num_hallucinated_links}")
    p = t.performance
    print(f"`-- performance: retrieval={p.retrieval_ms:.0f} generation={p.generation_ms:.0f} "
          f"evaluation={p.evaluation_ms:.0f} total={p.total_ms:.0f} ms  "
          f"tokens={p.token_usage.total_tokens}  cost=${p.estimated_cost_usd:.4f}")
    for err in t.errors:
        print(f"    ERROR: {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage tracked experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(func=cmd_list)

    for name, func in (("show", cmd_show), ("metrics", cmd_metrics), ("delete", cmd_delete)):
        p = sub.add_parser(name)
        p.add_argument("experiment_id")
        p.set_defaults(func=func)

    p_trace = sub.add_parser("trace")
    p_trace.add_argument("experiment_id")
    p_trace.add_argument("--question", help="Show only this question's trace.")
    p_trace.add_argument("--all", action="store_true", help="Show every question's trace.")
    p_trace.set_defaults(func=cmd_trace)

    args = parser.parse_args()
    with ExperimentStore() as store:
        args.func(store, args)


if __name__ == "__main__":
    main()
