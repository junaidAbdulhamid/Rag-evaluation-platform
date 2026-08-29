import pytest

from app.models import Chunk
from app.retrieval.vector_store import InMemoryVectorStore


def chunk(cid: str) -> Chunk:
    return Chunk(chunk_id=cid, document_id="doc", text=cid)


def test_empty_store_returns_no_results():
    store = InMemoryVectorStore()

    assert len(store) == 0
    assert store.search([1.0, 0.0, 0.0], top_k=3) == []


def test_search_orders_by_cosine_similarity():
    store = InMemoryVectorStore()
    store.add(
        [chunk("A"), chunk("B"), chunk("C")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )

    results = store.search([0.9, 0.1, 0.0], top_k=3)

    assert [c.chunk_id for c, _ in results] == ["A", "B", "C"]
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0.99  # query is nearly parallel to A
    assert scores[2] == pytest.approx(0.0, abs=1e-6)  # query is orthogonal to C


def test_search_is_capped_at_number_of_stored_chunks():
    store = InMemoryVectorStore()
    store.add([chunk("A"), chunk("B")], [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 1.0], top_k=10)

    assert len(results) == 2


def test_add_accumulates_across_calls():
    store = InMemoryVectorStore()
    store.add([chunk("A")], [[1.0, 0.0]])
    store.add([chunk("B")], [[0.0, 1.0]])

    assert len(store) == 2
    assert {c.chunk_id for c, _ in store.search([1.0, 1.0], top_k=2)} == {"A", "B"}


def test_add_rejects_length_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([chunk("A"), chunk("B")], [[1.0, 0.0]])


def test_search_rejects_non_positive_top_k():
    store = InMemoryVectorStore()
    store.add([chunk("A")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], top_k=0)
