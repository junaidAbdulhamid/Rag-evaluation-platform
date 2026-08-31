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

    args = parser.parse_args()
    with ExperimentStore() as store:
        args.func(store, args)


if __name__ == "__main__":
    main()
