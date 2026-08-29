from app.models import Chunk
from app.retrieval.retriever import DenseRetriever
from app.retrieval.vector_store import InMemoryVectorStore
from tests.fakes import FakeEmbeddingProvider


def build_retriever() -> DenseRetriever:
    # FakeEmbeddingProvider is bag-of-words, so tests share *exact* keywords
    # ("refund", "request") between the query and the intended target chunk.
    embeddings = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    chunks = [
        Chunk(chunk_id="refund::0", document_id="refund", text="You can request a refund within 30 days."),
        Chunk(chunk_id="shipping::0", document_id="shipping", text="Standard shipping takes three to five business days."),
        Chunk(chunk_id="warranty::0", document_id="warranty", text="The limited warranty lasts twelve months."),
    ]
    store.add(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return DenseRetriever(embeddings, store)


def test_retrieve_ranks_relevant_chunk_first_and_numbers_ranks():
    retriever = build_retriever()

    results = retriever.retrieve("How do I request a refund?", top_k=3)

    assert results[0].chunk.document_id == "refund"
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].score >= results[1].score >= results[2].score


def test_retrieve_respects_top_k():
    retriever = build_retriever()

    results = retriever.retrieve("refund", top_k=1)

    assert len(results) == 1
    assert results[0].rank == 1
    assert results[0].chunk.document_id == "refund"
