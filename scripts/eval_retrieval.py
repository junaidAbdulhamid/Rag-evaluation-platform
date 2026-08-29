"""Phase 3 CLI: evaluate retrieval quality over the whole golden dataset.

    python -m scripts.eval_retrieval
    python -m scripts.eval_retrieval --top-k 3 --chunk-size 300 --chunk-overlap 30
    python -m scripts.eval_retrieval --worst 8

Prints dataset-level metrics and the worst-performing questions, so you can see
*which* queries retrieval is failing on - not just the averages.
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.evaluation.dataset import load_eval_dataset
from app.evaluation.retrieval import RetrievalEvaluation, evaluate_retrieval
from app.ingestion.chunker import TextChunker
from app.ingestion.embeddings import SentenceTransformerEmbeddingProvider
from app.pipeline import RagPipeline
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore

RULE = "=" * 78


def build_retrieval_pipeline(args: argparse.Namespace) -> RagPipeline:
    """A pipeline with a real generator slot we never call - retrieval only."""
    embeddings = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    vector_store = InMemoryVectorStore()
    return RagPipeline(
        chunker=TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=DenseRetriever(embeddings, vector_store),
        generator=None,  # retrieval-only pipeline
        default_top_k=args.top_k,
    )


def print_report(evaluation: RetrievalEvaluation, worst_n: int) -> None:
    agg = evaluation.aggregate
    print(f"\n{RULE}\nRETRIEVAL METRICS  @k={agg.k}\n{RULE}")
    print(f"questions: {agg.num_questions_total} total, {agg.num_questions_scored} scored "
          f"(unanswerable excluded)")
    print(f"  Hit@{agg.k}       {agg.hit_rate:.3f}")
    print(f"  Precision@{agg.k} {agg.precision:.3f}")
    print(f"  Recall@{agg.k}    {agg.recall:.3f}")
    print(f"  MRR          {agg.mrr:.3f}")
    print(f"  NDCG@{agg.k}     {agg.ndcg:.3f}")

    answerable = [r for r in evaluation.per_question if not r.is_unanswerable]
    # rank by (hit, recall, reciprocal_rank) ascending - worst first
    worst = sorted(
        answerable,
        key=lambda r: (r.metrics.hit_rate, r.metrics.recall or 0.0, r.metrics.reciprocal_rank),
    )[:worst_n]

    print(f"\n{RULE}\nWORST {len(worst)} QUESTIONS\n{RULE}")
    for r in worst:
        m = r.metrics
        print(f"\n{r.question_id}  hit={m.hit_rate:.0f} recall={m.recall:.2f} "
              f"rr={m.reciprocal_rank:.2f} ndcg={m.ndcg:.2f}")
        print(f"  relevant : {r.relevant_doc_ids}")
        print(f"  retrieved: {r.retrieved_doc_ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval over the golden dataset.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--worst", type=int, default=5, help="How many worst questions to list.")
    args = parser.parse_args()

    pipeline = build_retrieval_pipeline(args)
    print("Ingesting corpus...")
    summary = pipeline.ingest(settings.documents_dir)
    print(f"  {summary.document_count} docs -> {summary.chunk_count} chunks")

    dataset = load_eval_dataset()
    evaluation = evaluate_retrieval(dataset, pipeline.retrieve, k=args.top_k)
    print_report(evaluation, worst_n=args.worst)


if __name__ == "__main__":
    main()
