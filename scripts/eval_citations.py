"""Phase 6 CLI: generate cited answers and evaluate their citations.

    python -m scripts.eval_citations --limit 5
    python -m scripts.eval_citations

COST: per question, 1 API call to generate the cited answer + 1 to evaluate the
citations. Use --limit while iterating.
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.evaluation.citation import LLMCitationEvaluator
from app.evaluation.citation_eval import CitationEvaluation, evaluate_citations
from app.evaluation.dataset import load_eval_dataset
from app.generation.citation import AnthropicCitedGenerator
from app.llm import AnthropicTextLLM
from app.pipeline import build_default_pipeline

RULE = "=" * 78


def print_report(evaluation: CitationEvaluation, worst_n: int) -> None:
    agg = evaluation.aggregate
    print(f"\n{RULE}\nCITATION METRICS  ({agg.num_questions} questions, {agg.num_scored} with claims)\n{RULE}")
    print(f"  completeness        {agg.citation_completeness:.3f}   (claims that carry a citation)")
    print(f"  precision           {agg.citation_precision:.3f}   (citation links that support their claim)")
    print(f"  correctness         {agg.citation_correctness:.3f}   (cited claims backed by a valid source)")
    print(f"  hallucination_rate  {agg.citation_hallucination_rate:.3f}   "
          f"({agg.total_hallucinated_links}/{agg.total_links} links point nowhere)")

    worst = sorted(
        (r for r in evaluation.per_question if r.has_claims),
        key=lambda r: (
            r.result.citation_precision if r.result.citation_precision is not None else 1.0,
            r.result.citation_completeness or 0.0,
        ),
    )[:worst_n]

    print(f"\n{RULE}\nWORST {len(worst)} ANSWERS\n{RULE}")
    for r in worst:
        res = r.result
        print(f"\n{r.question_id}  completeness={res.citation_completeness:.2f} "
              f"precision={res.citation_precision if res.citation_precision is not None else '-'} "
              f"halluc={res.num_hallucinated_links}")
        print(f"  answer: {r.answer}")
        for claim in res.claims:
            tag = f"cites {claim.markers}" if claim.has_citation else "NO CITATION"
            print(f"    - {claim.text}  [{tag}]")
        for link in res.links:
            state = "hallucinated" if not link.exists else ("supported" if link.supports_claim else "unsupported")
            print(f"      [{link.marker}] -> {link.resolved_chunk_id or '(none)'}  {state}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate citation grounding over the golden dataset.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    pipeline = build_default_pipeline()
    print("Ingesting corpus...")
    pipeline.ingest(settings.documents_dir)

    dataset = load_eval_dataset()
    examples = list(dataset)[: args.limit] if args.limit else list(dataset)

    generator = AnthropicCitedGenerator(
        model=settings.generation_model, api_key=settings.anthropic_api_key
    )
    evaluator = LLMCitationEvaluator(
        AnthropicTextLLM(model=settings.generation_model, api_key=settings.anthropic_api_key)
    )

    print(f"Generating cited answers for {len(examples)} questions...")
    cases = []
    for example in examples:
        retrieved = pipeline.retrieve(example.question, top_k=args.top_k)
        cited = generator.generate(example.question, retrieved)
        cases.append((example.id, cited.answer, retrieved))

    print("Evaluating citations...")
    evaluation = evaluate_citations(cases, evaluator)
    print_report(evaluation, worst_n=args.worst)


if __name__ == "__main__":
    main()
