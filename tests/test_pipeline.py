from pathlib import Path

from app.ingestion.chunker import TextChunker
from app.pipeline import RagPipeline
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from tests.fakes import EchoGenerator, FakeEmbeddingProvider


def build_pipeline(tmp_path: Path) -> tuple[RagPipeline, EchoGenerator]:
    (tmp_path / "refund_policy.md").write_text(
        "You can request a refund within 30 days of delivery.", encoding="utf-8"
    )
    (tmp_path / "shipping_policy.md").write_text(
        "Standard shipping takes three to five business days.", encoding="utf-8"
    )

    embeddings = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    generator = EchoGenerator()
    pipeline = RagPipeline(
        chunker=TextChunker(chunk_size=200, chunk_overlap=20),
        embeddings=embeddings,
        vector_store=store,
        retriever=DenseRetriever(embeddings, store),
        generator=generator,
        default_top_k=2,
    )
    return pipeline, generator


def test_ingest_reports_document_and_chunk_counts(tmp_path: Path):
    pipeline, _ = build_pipeline(tmp_path)

    summary = pipeline.ingest(str(tmp_path))

    assert summary.document_count == 2
    assert summary.chunk_count >= 2
    assert set(summary.document_ids) == {"refund_policy", "shipping_policy"}
    assert len(pipeline.vector_store) == summary.chunk_count


def test_answer_retrieves_context_then_calls_generator(tmp_path: Path):
    pipeline, generator = build_pipeline(tmp_path)
    pipeline.ingest(str(tmp_path))

    result = pipeline.answer("How do I request a refund?")

    # retrieval ran and is ranked
    assert result.retrieved_chunks
    assert result.retrieved_chunks[0].chunk.document_id == "refund_policy"
    assert [c.rank for c in result.retrieved_chunks] == list(
        range(1, len(result.retrieved_chunks) + 1)
    )
    # the generator received exactly what retrieval produced
    assert generator.calls[0][1] == result.retrieved_chunks
    assert "refund" in result.generated_answer.answer


def test_answer_top_k_override_beats_default(tmp_path: Path):
    pipeline, generator = build_pipeline(tmp_path)
    pipeline.ingest(str(tmp_path))

    pipeline.answer("shipping time?", top_k=1)

    assert len(generator.calls[-1][1]) == 1
