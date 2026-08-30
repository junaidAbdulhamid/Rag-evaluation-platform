"""Phase 5 CLI: evaluate answer faithfulness (grounding in retrieved context).

    python -m scripts.eval_faithfulness --limit 5
    python -m scripts.eval_faithfulness

COST: per question this makes 1 API call to answer, then up to 2 more (claim
extraction + verification). Use --limit while iterating.
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.evaluation.dataset import load_eval_dataset
from app.evaluation.faithfulness import LLMFaithfulnessEvaluator
from app.evaluation.faithfulness_eval import FaithfulnessEvaluation, evaluate_faithfulness
from app.llm import AnthropicTextLLM
from app.pipeline import build_default_pipeline

RULE = "=" * 78


def print_report(evaluation: FaithfulnessEvaluation, worst_n: int) -> None:
    agg = evaluation.aggregate
    print(f"\n{RULE}\nFAITHFULNESS  ({agg.num_questions} questions, {agg.num_scored} with claims)\n{RULE}")
    print(f"  faithfulness (macro)   {agg.faithfulness:.3f}")
    print(f"  claim_support_rate     {agg.claim_support_rate:.3f}  "
          f"({agg.total_supported}/{agg.total_claims} claims)")

    worst = sorted(
        (r for r in evaluation.per_question if r.has_claims),
        key=lambda r: (r.result.score, -r.result.num_claims),
    )[:worst_n]

    print(f"\n{RULE}\nLEAST FAITHFUL {len(worst)} ANSWERS\n{RULE}")
    for r in worst:
        print(f"\n{r.question_id}  score={r.result.score:.2f}  "
              f"({r.result.num_supported}/{r.result.num_claims} claims supported)")
        print(f"  answer: {r.answer}")
        for claim in r.result.claims:
            mark = "OK " if claim.supported else "!! "
            print(f"    {mark}{claim.text}")
            print(f"       -> {claim.reason}  {claim.supporting_chunk_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate faithfulness over the golden dataset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    print("Ingesting corpus...")
    pipeline.ingest(settings.documents_dir)

    dataset = load_eval_dataset()
    examples = list(dataset)[: args.limit] if args.limit else list(dataset)

    print(f"Answering {len(examples)} questions...")
    cases = []
    for example in examples:
        result = pipeline.answer(example.question, top_k=args.top_k)
        cases.append((example.id, result))

    print("Extracting and verifying claims...")
    evaluator = LLMFaithfulnessEvaluator(
        AnthropicTextLLM(model=settings.generation_model, api_key=settings.anthropic_api_key)
    )
    evaluation = evaluate_faithfulness(cases, evaluator)
    print_report(evaluation, worst_n=args.worst)


if __name__ == "__main__":
    main()
