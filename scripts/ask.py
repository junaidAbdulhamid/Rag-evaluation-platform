"""Phase 1 CLI: ingest the corpus, ask one question, inspect the result.

Run from the repo root:

    python -m scripts.ask --question "How many days do I have to request a refund?"

Optional flags let you feel how the knobs change retrieval:

    python -m scripts.ask -q "..." --top-k 3 --chunk-size 300 --chunk-overlap 30
    python -m scripts.ask -q "..." --show-prompt
"""

from __future__ import annotations

import argparse

from app.config import settings
from app.generation.generator import AnthropicGenerator
from app.ingestion.chunker import TextChunker
from app.ingestion.embeddings import SentenceTransformerEmbeddingProvider
from app.models import RagResult
from app.pipeline import RagPipeline
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore

RULE = "=" * 78


def build_pipeline(args: argparse.Namespace) -> RagPipeline:
    """Like `build_default_pipeline`, but lets CLI flags override chunking / top_k."""
    embeddings = SentenceTransformerEmbeddingProvider(settings.embedding_model_name)
    vector_store = InMemoryVectorStore()
    return RagPipeline(
        chunker=TextChunker(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
        embeddings=embeddings,
        vector_store=vector_store,
        retriever=DenseRetriever(embeddings, vector_store),
        generator=AnthropicGenerator(
            model=settings.generation_model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.generation_max_tokens,
        ),
        default_top_k=args.top_k,
    )


def print_result(result: RagResult, *, show_prompt: bool) -> None:
    print(f"\n{RULE}\nQUESTION\n{RULE}\n{result.question}")

    print(f"\n{RULE}\nRETRIEVED CHUNKS (top {len(result.retrieved_chunks)})\n{RULE}")
    for item in result.retrieved_chunks:
        preview = " ".join(item.chunk.text.split())[:220]
        print(f"\n#{item.rank}  score={item.score:.4f}  id={item.chunk.chunk_id}")
        print(f"    {preview}{'...' if len(item.chunk.text) > 220 else ''}")

    if show_prompt and result.generated_answer.prompt:
        print(f"\n{RULE}\nPROMPT SENT TO MODEL\n{RULE}\n{result.generated_answer.prompt}")

    ans = result.generated_answer
    print(f"\n{RULE}\nGENERATED ANSWER\n{RULE}\n{ans.answer}")

    print(f"\n{RULE}\nMETADATA\n{RULE}")
    print(f"model: {ans.model}")
    if ans.token_usage:
        u = ans.token_usage
        print(f"tokens: prompt={u.prompt_tokens} completion={u.completion_tokens} total={u.total_tokens}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask one question against the sample corpus.")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--docs", default=settings.documents_dir, help="Documents directory.")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--show-prompt", action="store_true", help="Also print the full prompt.")
    args = parser.parse_args()

    pipeline = build_pipeline(args)

    print("Ingesting corpus (first run downloads the embedding model)...")
    summary = pipeline.ingest(args.docs)
    print(
        f"  {summary.document_count} documents -> {summary.chunk_count} chunks "
        f"({', '.join(summary.document_ids)})"
    )

    result = pipeline.answer(args.question, top_k=args.top_k)
    print_result(result, show_prompt=args.show_prompt)


if __name__ == "__main__":
    main()
