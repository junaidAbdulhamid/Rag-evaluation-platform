"""Phase 4 CLI: evaluate generated answers over the golden dataset.

    python -m scripts.eval_generation --no-judge            # deterministic only
    python -m scripts.eval_generation --limit 5             # + LLM judge, first 5 Qs
    python -m scripts.eval_generation                        # + LLM judge, all Qs

COST: this calls the Anthropic API once per question to produce the answer, and
(without --no-judge) once more per question for the judge. Use --limit while
iterating.
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.evaluation.dataset import load_eval_dataset
from app.evaluation.generation import GenerationEvaluation, evaluate_generation
from app.evaluation.judge import LLMGenerationJudge
from app.llm import AnthropicTextLLM
from app.pipeline import build_default_pipeline

RULE = "=" * 78


def print_report(evaluation: GenerationEvaluation, worst_n: int) -> None:
    agg = evaluation.aggregate
    print(f"\n{RULE}\nGENERATION METRICS  ({agg.num_questions} questions)\n{RULE}")
    print("deterministic:")
    print(f"  exact_match          {agg.exact_match:.3f}")
    print(f"  token_f1             {agg.token_f1:.3f}")
    print(f"  token_recall         {agg.token_recall:.3f}")
    print(f"  number_coverage      {agg.number_coverage:.3f}  (n={agg.num_with_numbers})")
    print(f"  abstention_accuracy  {agg.abstention_accuracy:.3f}")
    if agg.num_judged:
        print(f"judge (n={agg.num_judged}):")
        print(f"  correctness          {agg.judge_correctness:.3f}")
        print(f"  relevance            {agg.judge_relevance:.3f}")

    def rank_key(r):
        if r.judgement is not None:
            return (r.judgement.correctness, r.judgement.relevance)
        return (r.deterministic.token_f1, r.deterministic.exact_match)

    worst = sorted(evaluation.per_question, key=rank_key)[:worst_n]
    print(f"\n{RULE}\nWORST {len(worst)} ANSWERS\n{RULE}")
    for r in worst:
        d = r.deterministic
        print(f"\n{r.question_id}  f1={d.token_f1:.2f} abst_ok={d.abstention_correct}", end="")
        if r.judgement is not None:
            print(f"  correctness={r.judgement.correctness:.2f} relevance={r.judgement.relevance:.2f}")
        else:
            print()
        print(f"  Q        : {r.question}")
        print(f"  expected : {r.expected_answer}")
        print(f"  generated: {r.generated_answer}")
        if r.judgement is not None:
            print(f"  why      : {r.judgement.correctness_reasoning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation over the golden dataset.")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--judge-model", default=settings.generation_model)
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    print("Ingesting corpus...")
    summary = pipeline.ingest(settings.documents_dir)
    print(f"  {summary.document_count} docs -> {summary.chunk_count} chunks")

    dataset = load_eval_dataset()
    examples = list(dataset)[: args.limit] if args.limit else list(dataset)

    print(f"Generating answers for {len(examples)} questions...")
    cases = []
    for example in examples:
        result = pipeline.answer(example.question, top_k=args.top_k)
        cases.append((example, result.generated_answer.answer))

    judge = None
    if not args.no_judge:
        print("Judging answers...")
        judge = LLMGenerationJudge(AnthropicTextLLM(model=args.judge_model, api_key=settings.anthropic_api_key))

    evaluation = evaluate_generation(cases, judge=judge)
    print_report(evaluation, worst_n=args.worst)


if __name__ == "__main__":
    main()
