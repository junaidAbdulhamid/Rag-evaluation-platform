import pytest

from app.ingestion.chunker import TextChunker
from app.models import Document


def make_doc(text: str, document_id: str = "doc") -> Document:
    return Document(document_id=document_id, filename=f"{document_id}.txt", text=text)


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=-1)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)  # overlap must be < size


def test_sliding_window_size_overlap_and_ids():
    text = "0123456789ABCDEFGHIJ"  # 20 chars
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)  # step = 8

    chunks = chunker.chunk_document(make_doc(text))

    assert [c.text for c in chunks] == ["0123456789", "89ABCDEFGH", "GHIJ"]
    assert [c.chunk_id for c in chunks] == ["doc::chunk_0", "doc::chunk_1", "doc::chunk_2"]
    # consecutive chunks overlap by exactly `chunk_overlap` characters
    assert chunks[0].text[-2:] == chunks[1].text[:2]
    assert chunks[1].metadata["char_start"] == 8
    assert all(c.document_id == "doc" for c in chunks)


def test_text_shorter_than_chunk_size_yields_one_chunk():
    chunks = TextChunker(chunk_size=100, chunk_overlap=10).chunk_document(make_doc("short text"))

    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_empty_and_whitespace_only_text_yield_no_chunks():
    chunker = TextChunker(chunk_size=10, chunk_overlap=2)

    assert chunker.chunk_document(make_doc("")) == []
    assert chunker.chunk_document(make_doc("      ")) == []


def test_chunk_documents_flattens_and_keeps_per_document_ids():
    chunker = TextChunker(chunk_size=6, chunk_overlap=1)  # step = 5
    docs = [make_doc("aaaaaaaa", "a"), make_doc("bbbbbbbb", "b")]

    chunks = chunker.chunk_documents(docs)

    assert {c.document_id for c in chunks} == {"a", "b"}
    assert chunks[0].chunk_id == "a::chunk_0"
    assert any(c.chunk_id == "b::chunk_0" for c in chunks)
